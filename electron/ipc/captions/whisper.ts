import { createWriteStream } from "node:fs";
import { constants as fsConstants } from "node:fs";
import fs from "node:fs/promises";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { ReadableStream as NodeReadableStream } from "node:stream/web";
import type Electron from "electron";
import { session } from "electron";
import {
	WHISPER_MODEL_DIR,
	WHISPER_MODEL_DOWNLOAD_URL,
	WHISPER_SMALL_MODEL_PATH,
} from "../constants";

export function sendWhisperModelDownloadProgress(
	webContents: Electron.WebContents,
	payload: {
		status: "idle" | "downloading" | "downloaded" | "error";
		progress: number;
		path?: string | null;
		error?: string;
	},
) {
	webContents.send("whisper-small-model-download-progress", payload);
}

export async function getWhisperSmallModelStatus() {
	try {
		await fs.access(WHISPER_SMALL_MODEL_PATH, fsConstants.R_OK);
		return {
			success: true,
			exists: true,
			path: WHISPER_SMALL_MODEL_PATH,
		};
	} catch {
		return {
			success: true,
			exists: false,
			path: null,
		};
	}
}

const DOWNLOAD_IDLE_TIMEOUT_MS = 30_000;
const PROXY_ENV_KEYS = [
	"HTTPS_PROXY",
	"https_proxy",
	"HTTP_PROXY",
	"http_proxy",
	"ALL_PROXY",
	"all_proxy",
];

/** Reads the proxy the user configured through the usual environment variables. */
export function getProxyConfigFromEnv(
	env: NodeJS.ProcessEnv = process.env,
): Electron.ProxyConfig | null {
	const proxyRules = PROXY_ENV_KEYS.map((key) => env[key]?.trim()).find(Boolean);
	if (!proxyRules) {
		return null;
	}

	const proxyBypassRules = (env.NO_PROXY ?? env.no_proxy)?.trim();
	return proxyBypassRules ? { proxyRules, proxyBypassRules } : { proxyRules };
}

// Electron's network stack follows the system proxy settings (unlike node:https).
// A proxy set through environment variables wins, as it does for curl and wget.
async function getDownloadSession(): Promise<Electron.Session> {
	const proxyConfig = getProxyConfigFromEnv();
	if (!proxyConfig) {
		return session.defaultSession;
	}

	const downloadSession = session.fromPartition("recordly-downloads");
	await downloadSession.setProxy(proxyConfig);
	return downloadSession;
}

export async function downloadFileWithProgress(
	url: string,
	destinationPath: string,
	onProgress: (progress: number) => void,
): Promise<void> {
	const downloadSession = await getDownloadSession();
	const controller = new AbortController();
	let timedOut = false;
	let idleTimer: NodeJS.Timeout | undefined;
	const resetIdleTimer = () => {
		clearTimeout(idleTimer);
		idleTimer = setTimeout(() => {
			timedOut = true;
			controller.abort();
		}, DOWNLOAD_IDLE_TIMEOUT_MS);
	};

	resetIdleTimer();
	try {
		const response = await downloadSession.fetch(url, { signal: controller.signal });
		if (!response.ok || !response.body) {
			throw new Error(`Whisper model download failed with status ${response.status}.`);
		}

		const totalBytes = Number.parseInt(response.headers.get("content-length") ?? "0", 10);
		let downloadedBytes = 0;
		let lastProgress = -1;

		await pipeline(
			Readable.fromWeb(response.body as NodeReadableStream<Uint8Array>),
			async function* (chunks: AsyncIterable<Buffer>) {
				for await (const chunk of chunks) {
					resetIdleTimer();
					downloadedBytes += chunk.length;
					if (Number.isFinite(totalBytes) && totalBytes > 0) {
						const progress = Math.min(
							100,
							Math.round((downloadedBytes / totalBytes) * 100),
						);
						if (progress !== lastProgress) {
							lastProgress = progress;
							onProgress(progress);
						}
					}
					yield chunk;
				}
			},
			createWriteStream(destinationPath),
		);
		if (lastProgress !== 100) {
			onProgress(100);
		}
	} catch (error) {
		if (timedOut) {
			throw new Error("Whisper model download timed out.");
		}
		throw error;
	} finally {
		clearTimeout(idleTimer);
	}
}

export async function downloadWhisperSmallModel(
	webContents: Electron.WebContents,
): Promise<string> {
	await fs.mkdir(WHISPER_MODEL_DIR, { recursive: true });
	const tempPath = `${WHISPER_SMALL_MODEL_PATH}.download`;

	sendWhisperModelDownloadProgress(webContents, {
		status: "downloading",
		progress: 0,
		path: null,
	});

	try {
		await fs.rm(tempPath, { force: true });
		await downloadFileWithProgress(WHISPER_MODEL_DOWNLOAD_URL, tempPath, (progress) => {
			sendWhisperModelDownloadProgress(webContents, {
				status: "downloading",
				progress,
				path: null,
			});
		});
		await fs.rename(tempPath, WHISPER_SMALL_MODEL_PATH);
		sendWhisperModelDownloadProgress(webContents, {
			status: "downloaded",
			progress: 100,
			path: WHISPER_SMALL_MODEL_PATH,
		});
		return WHISPER_SMALL_MODEL_PATH;
	} catch (error) {
		await fs.rm(tempPath, { force: true }).catch(() => undefined);
		sendWhisperModelDownloadProgress(webContents, {
			status: "error",
			progress: 0,
			path: null,
			error: String(error),
		});
		throw error;
	}
}

export async function deleteWhisperSmallModel(): Promise<void> {
	await fs.rm(WHISPER_SMALL_MODEL_PATH, { force: true });
}
