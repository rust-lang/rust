import * as assert from "assert";
import type { Context } from ".";
import { substituteVariablesInEnv } from "../../src/config";

export async function getTests(ctx: Context) {
    await ctx.suite("Server Env Settings", (suite) => {
        suite.addTest("Replacing Env Variables", async () => {
            const envJson = {
                USING_MY_VAR: "${env:MY_VAR} test ${env:MY_VAR}",
                MY_VAR: "test",
            };
            const expectedEnv = {
                USING_MY_VAR: "test test test",
                MY_VAR: "test",
            };
            const actualEnv = await substituteVariablesInEnv(envJson);
            assert.deepStrictEqual(actualEnv, expectedEnv);
        });

        suite.addTest("Circular dependencies remain as is", async () => {
            const envJson = {
                A_USES_B: "${env:B_USES_A}",
                B_USES_A: "${env:A_USES_B}",
                C_USES_ITSELF: "${env:C_USES_ITSELF}",
                D_USES_C: "${env:C_USES_ITSELF}",
                E_IS_ISOLATED: "test",
                F_USES_E: "${env:E_IS_ISOLATED}",
            };
            const expectedEnv = {
                A_USES_B: "${env:B_USES_A}",
                B_USES_A: "${env:A_USES_B}",
                C_USES_ITSELF: "${env:C_USES_ITSELF}",
                D_USES_C: "${env:C_USES_ITSELF}",
                E_IS_ISOLATED: "test",
                F_USES_E: "test",
            };
            const actualEnv = await substituteVariablesInEnv(envJson);
            assert.deepStrictEqual(actualEnv, expectedEnv);
        });

        suite.addTest("Should support external variables", async () => {
            process.env["TEST_VARIABLE"] = "test";
            const envJson = {
                USING_EXTERNAL_VAR: "${env:TEST_VARIABLE} test ${env:TEST_VARIABLE}",
            };
            const expectedEnv = {
                USING_EXTERNAL_VAR: "test test test",
            };

            const actualEnv = await substituteVariablesInEnv(envJson);
            assert.deepStrictEqual(actualEnv, expectedEnv);
            delete process.env["TEST_VARIABLE"];
        });

        suite.addTest("should support VSCode variables", async () => {
            const envJson = {
                USING_VSCODE_VAR: "${workspaceFolderBasename}",
            };
            const actualEnv = await substituteVariablesInEnv(envJson);
            assert.deepStrictEqual(actualEnv["USING_VSCODE_VAR"], "code");
        });
    });
}
