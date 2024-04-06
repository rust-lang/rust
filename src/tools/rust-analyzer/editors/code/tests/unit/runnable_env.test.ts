import * as assert from "assert";
import { prepareEnv } from "../../src/run";
import type { RunnableEnvCfg } from "../../src/config";
import type { Context } from ".";
import type * as ra from "../../src/lsp_ext";

function makeRunnable(label: string): ra.Runnable {
    return {
        label,
        kind: "cargo",
        args: {
            cargoArgs: [],
            executableArgs: [],
            cargoExtraArgs: [],
        },
    };
}

function fakePrepareEnv(runnableName: string, config: RunnableEnvCfg): Record<string, string> {
    const runnable = makeRunnable(runnableName);
    return prepareEnv(runnable, config);
}

export async function getTests(ctx: Context) {
    await ctx.suite("Runnable env", (suite) => {
        suite.addTest("Global config works", async () => {
            const binEnv = fakePrepareEnv("run project_name", { GLOBAL: "g" });
            assert.strictEqual(binEnv["GLOBAL"], "g");

            const testEnv = fakePrepareEnv("test some::mod::test_name", { GLOBAL: "g" });
            assert.strictEqual(testEnv["GLOBAL"], "g");
        });

        suite.addTest("null mask works", async () => {
            const config = [
                {
                    env: { DATA: "data" },
                },
            ];
            const binEnv = fakePrepareEnv("run project_name", config);
            assert.strictEqual(binEnv["DATA"], "data");

            const testEnv = fakePrepareEnv("test some::mod::test_name", config);
            assert.strictEqual(testEnv["DATA"], "data");
        });

        suite.addTest("order works", async () => {
            const config = [
                {
                    env: { DATA: "data" },
                },
                {
                    env: { DATA: "newdata" },
                },
            ];
            const binEnv = fakePrepareEnv("run project_name", config);
            assert.strictEqual(binEnv["DATA"], "newdata");

            const testEnv = fakePrepareEnv("test some::mod::test_name", config);
            assert.strictEqual(testEnv["DATA"], "newdata");
        });

        suite.addTest("mask works", async () => {
            const config = [
                {
                    env: { DATA: "data" },
                },
                {
                    mask: "^run",
                    env: { DATA: "rundata" },
                },
                {
                    mask: "special_test$",
                    env: { DATA: "special_test" },
                },
            ];
            const binEnv = fakePrepareEnv("run project_name", config);
            assert.strictEqual(binEnv["DATA"], "rundata");

            const testEnv = fakePrepareEnv("test some::mod::test_name", config);
            assert.strictEqual(testEnv["DATA"], "data");

            const specialTestEnv = fakePrepareEnv("test some::mod::special_test", config);
            assert.strictEqual(specialTestEnv["DATA"], "special_test");
        });

        suite.addTest("exact test name works", async () => {
            const config = [
                {
                    env: { DATA: "data" },
                },
                {
                    mask: "some::mod::test_name",
                    env: { DATA: "test special" },
                },
            ];
            const testEnv = fakePrepareEnv("test some::mod::test_name", config);
            assert.strictEqual(testEnv["DATA"], "test special");

            const specialTestEnv = fakePrepareEnv("test some::mod::another_test", config);
            assert.strictEqual(specialTestEnv["DATA"], "data");
        });

        suite.addTest("test mod name works", async () => {
            const config = [
                {
                    env: { DATA: "data" },
                },
                {
                    mask: "some::mod",
                    env: { DATA: "mod special" },
                },
            ];
            const testEnv = fakePrepareEnv("test some::mod::test_name", config);
            assert.strictEqual(testEnv["DATA"], "mod special");

            const specialTestEnv = fakePrepareEnv("test some::mod::another_test", config);
            assert.strictEqual(specialTestEnv["DATA"], "mod special");
        });
    });
}
