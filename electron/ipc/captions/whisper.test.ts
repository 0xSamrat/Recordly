import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const electronMocks = vi.hoisted(() => {
	const defaultFetch = vi.fn();
	const proxyFetch = vi.fn();
	const setProxy = vi.fn(async () => undefined);
	const fromPartition = vi.fn(() => ({ fetch: proxyFetch, setProxy }));
	return { defaultFetch, proxyFetch, setProxy, fromPartition };
});

vi.mock("electron", () => ({
	session: {
		defaultSession: { fetch: electronMocks.defaultFetch },
		fromPartition: electronMocks.fromPartition,
	},
}));
vi.mock("../constants", () => ({
	WHISPER_MODEL_DIR: "/tmp/whisper",
	WHISPER_MODEL_DOWNLOAD_URL: "https://example.com/ggml-small.bin",
	WHISPER_SMALL_MODEL_PATH: "/tmp/whisper/ggml-small.bin",
}));

import { downloadFileWithProgress, getProxyConfigFromEnv } from "./whisper";

const PROXY_ENV_KEYS = [
	"HTTPS_PROXY",
	"https_proxy",
	"HTTP_PROXY",
	"http_proxy",
	"ALL_PROXY",
	"all_proxy",
	"NO_PROXY",
	"no_proxy",
];

function chunkedResponse(chunks: string[], status = 200) {
	const encoder = new TextEncoder();
	const totalBytes = chunks.reduce((sum, chunk) => sum + encoder.encode(chunk).length, 0);
	const body = new ReadableStream<Uint8Array>({
		start(controller) {
			for (const chunk of chunks) {
				controller.enqueue(encoder.encode(chunk));
			}
			controller.close();
		},
	});
	return new Response(body, { status, headers: { "content-length": String(totalBytes) } });
}

describe("getProxyConfigFromEnv", () => {
	it("returns null when no proxy is configured", () => {
		expect(getProxyConfigFromEnv({})).toBeNull();
		expect(getProxyConfigFromEnv({ HTTPS_PROXY: "  " })).toBeNull();
	});

	it("prefers HTTPS_PROXY and accepts lowercase variables", () => {
		expect(
			getProxyConfigFromEnv({
				HTTP_PROXY: "http://http-proxy:3128",
				HTTPS_PROXY: "http://https-proxy:3128",
			}),
		).toEqual({ proxyRules: "http://https-proxy:3128" });
		expect(getProxyConfigFromEnv({ http_proxy: "http://127.0.0.1:7890" })).toEqual({
			proxyRules: "http://127.0.0.1:7890",
		});
	});

	it("passes NO_PROXY through as bypass rules", () => {
		expect(
			getProxyConfigFromEnv({
				ALL_PROXY: "socks5://127.0.0.1:1080",
				NO_PROXY: "localhost,.lan",
			}),
		).toEqual({ proxyRules: "socks5://127.0.0.1:1080", proxyBypassRules: "localhost,.lan" });
	});
});

describe("downloadFileWithProgress", () => {
	let tempRoot: string;
	let savedEnv: Record<string, string | undefined>;

	beforeEach(async () => {
		tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "recordly-whisper-download-"));
		savedEnv = Object.fromEntries(PROXY_ENV_KEYS.map((key) => [key, process.env[key]]));
		for (const key of PROXY_ENV_KEYS) {
			delete process.env[key];
		}
		vi.clearAllMocks();
	});

	afterEach(async () => {
		for (const [key, value] of Object.entries(savedEnv)) {
			if (value === undefined) {
				delete process.env[key];
			} else {
				process.env[key] = value;
			}
		}
		await fs.rm(tempRoot, { recursive: true, force: true });
	});

	it("downloads through Electron's default session so system proxy settings apply", async () => {
		electronMocks.defaultFetch.mockResolvedValue(chunkedResponse(["ab", "cd", "ef", "gh"]));
		const destinationPath = path.join(tempRoot, "model.bin");
		const progress: number[] = [];

		await downloadFileWithProgress("https://example.com/model", destinationPath, (value) =>
			progress.push(value),
		);

		expect(electronMocks.defaultFetch).toHaveBeenCalledWith(
			"https://example.com/model",
			expect.objectContaining({ signal: expect.any(AbortSignal) }),
		);
		expect(electronMocks.fromPartition).not.toHaveBeenCalled();
		await expect(fs.readFile(destinationPath, "utf8")).resolves.toBe("abcdefgh");
		// Streams may merge chunks, so only check the progress contract.
		expect(progress.at(-1)).toBe(100);
		expect(progress).toEqual([...progress].sort((a, b) => a - b));
		expect(new Set(progress).size).toBe(progress.length);
	});

	it("uses the proxy from the environment when one is set", async () => {
		process.env.HTTPS_PROXY = "http://127.0.0.1:8899";
		electronMocks.proxyFetch.mockResolvedValue(chunkedResponse(["model"]));
		const destinationPath = path.join(tempRoot, "model.bin");

		await downloadFileWithProgress(
			"https://example.com/model",
			destinationPath,
			() => undefined,
		);

		expect(electronMocks.setProxy).toHaveBeenCalledWith({
			proxyRules: "http://127.0.0.1:8899",
		});
		expect(electronMocks.proxyFetch).toHaveBeenCalledTimes(1);
		expect(electronMocks.defaultFetch).not.toHaveBeenCalled();
		await expect(fs.readFile(destinationPath, "utf8")).resolves.toBe("model");
	});

	it("rejects when the server does not return the file", async () => {
		electronMocks.defaultFetch.mockResolvedValue(chunkedResponse(["Not Found"], 404));

		await expect(
			downloadFileWithProgress(
				"https://example.com/model",
				path.join(tempRoot, "model.bin"),
				() => undefined,
			),
		).rejects.toThrow("Whisper model download failed with status 404.");
	});
});
