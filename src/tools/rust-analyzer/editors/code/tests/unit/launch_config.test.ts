import * as assert from "assert";
import { Cargo } from "../../src/toolchain";
import type { Context } from ".";

export async function getTests(ctx: Context) {
    await ctx.suite("Launch configuration/Lens", (suite) => {
        suite.addTest("A binary", async () => {
            const args = Cargo.artifactSpec([
                "build",
                "--package",
                "pkg_name",
                "--bin",
                "pkg_name",
            ]);

            assert.deepStrictEqual(args.cargoArgs, [
                "build",
                "--package",
                "pkg_name",
                "--bin",
                "pkg_name",
                "--message-format=json",
            ]);
            assert.deepStrictEqual(args.filter, undefined);
        });

        suite.addTest("One of Multiple Binaries", async () => {
            const args = Cargo.artifactSpec(["build", "--package", "pkg_name", "--bin", "bin1"]);

            assert.deepStrictEqual(args.cargoArgs, [
                "build",
                "--package",
                "pkg_name",
                "--bin",
                "bin1",
                "--message-format=json",
            ]);
            assert.deepStrictEqual(args.filter, undefined);
        });

        suite.addTest("A test", async () => {
            const args = Cargo.artifactSpec(["test", "--package", "pkg_name", "--lib", "--no-run"]);

            assert.deepStrictEqual(args.cargoArgs, [
                "test",
                "--package",
                "pkg_name",
                "--lib",
                "--no-run",
                "--message-format=json",
            ]);
            assert.notDeepStrictEqual(args.filter, undefined);
        });
    });

    await ctx.suite("Launch configuration/QuickPick", (suite) => {
        suite.addTest("A binary", async () => {
            const args = Cargo.artifactSpec(["run", "--package", "pkg_name", "--bin", "pkg_name"]);

            assert.deepStrictEqual(args.cargoArgs, [
                "build",
                "--package",
                "pkg_name",
                "--bin",
                "pkg_name",
                "--message-format=json",
            ]);
            assert.deepStrictEqual(args.filter, undefined);
        });

        suite.addTest("One of Multiple Binaries", async () => {
            const args = Cargo.artifactSpec(["run", "--package", "pkg_name", "--bin", "bin2"]);

            assert.deepStrictEqual(args.cargoArgs, [
                "build",
                "--package",
                "pkg_name",
                "--bin",
                "bin2",
                "--message-format=json",
            ]);
            assert.deepStrictEqual(args.filter, undefined);
        });

        suite.addTest("A test", async () => {
            const args = Cargo.artifactSpec(["test", "--package", "pkg_name", "--lib"]);

            assert.deepStrictEqual(args.cargoArgs, [
                "test",
                "--package",
                "pkg_name",
                "--lib",
                "--message-format=json",
                "--no-run",
            ]);
            assert.notDeepStrictEqual(args.filter, undefined);
        });
    });
}
