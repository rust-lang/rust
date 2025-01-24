import * as assert from "assert";
import { _private } from "../../src/bootstrap";
import type { Context } from ".";

export async function getTests(ctx: Context) {
    await ctx.suite("Bootstrap/Select toolchain RA", (suite) => {
        suite.addTest("Order of nightly RA", async () => {
            assert.deepStrictEqual(
                await _private.orderFromPath(
                    "/Users/myuser/.rustup/toolchains/nightly-2022-11-22-aarch64-apple-darwin/bin/rust-analyzer",
                    async function (path: string) {
                        assert.deepStrictEqual(
                            path,
                            "/Users/myuser/.rustup/toolchains/nightly-2022-11-22-aarch64-apple-darwin/bin/rust-analyzer",
                        );
                        return "rust-analyzer 1.67.0-nightly (b7bc90fe 2022-11-21)";
                    },
                ),
                "0-2022-11-21/0",
            );
        });

        suite.addTest("Order of versioned RA", async () => {
            assert.deepStrictEqual(
                await _private.orderFromPath(
                    "/Users/myuser/.rustup/toolchains/1.72.1-aarch64-apple-darwin/bin/rust-analyzer",
                    async function (path: string) {
                        assert.deepStrictEqual(
                            path,
                            "/Users/myuser/.rustup/toolchains/1.72.1-aarch64-apple-darwin/bin/rust-analyzer",
                        );
                        return "rust-analyzer 1.72.1 (d5c2e9c3 2023-09-13)";
                    },
                ),
                "0-2023-09-13/1",
            );
        });

        suite.addTest("Order of versioned RA when unable to obtain version date", async () => {
            assert.deepStrictEqual(
                await _private.orderFromPath(
                    "/Users/myuser/.rustup/toolchains/1.72.1-aarch64-apple-darwin/bin/rust-analyzer",
                    async function () {
                        return "rust-analyzer 1.72.1";
                    },
                ),
                "2",
            );
        });

        suite.addTest("Order of stable RA", async () => {
            assert.deepStrictEqual(
                await _private.orderFromPath(
                    "/Users/myuser/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rust-analyzer",
                    async function (path: string) {
                        assert.deepStrictEqual(
                            path,
                            "/Users/myuser/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rust-analyzer",
                        );
                        return "rust-analyzer 1.79.0 (129f3b99 2024-06-10)";
                    },
                ),
                "0-2024-06-10/1",
            );
        });

        suite.addTest("Order with invalid path to RA", async () => {
            assert.deepStrictEqual(
                await _private.orderFromPath("some-weird-path", async function () {
                    return undefined;
                }),
                "2",
            );
        });

        suite.addTest("Earliest RA between nightly and stable", async () => {
            assert.deepStrictEqual(
                await _private.earliestToolchainPath(
                    "/Users/myuser/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rust-analyzer",
                    "/Users/myuser/.rustup/toolchains/nightly-2022-11-22-aarch64-apple-darwin/bin/rust-analyzer",
                    async function (path: string) {
                        if (
                            path ===
                            "/Users/myuser/.rustup/toolchains/nightly-2022-11-22-aarch64-apple-darwin/bin/rust-analyzer"
                        ) {
                            return "rust-analyzer 1.67.0-nightly (b7bc90fe 2022-11-21)";
                        } else {
                            return "rust-analyzer 1.79.0 (129f3b99 2024-06-10)";
                        }
                    },
                ),
                "/Users/myuser/.rustup/toolchains/nightly-2022-11-22-aarch64-apple-darwin/bin/rust-analyzer",
            );
        });
    });
}
