import * as assert from "assert";
import * as os from "os";
import * as path from "path";
import { mkdtemp, mkdir, rm, writeFile } from "fs/promises";
import { Cargo, cargoPath } from "../../src/toolchain";
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

    await ctx.suite("Toolchain resolution", (suite) => {
        suite.addTest("prefers explicit CARGO from provided env", async () => {
            const explicitCargo = path.join(os.tmpdir(), "custom-cargo");
            assert.strictEqual(await cargoPath({ CARGO: explicitCargo }), explicitCargo);
        });

        suite.addTest("resolves cargo from provided PATH", async () => {
            const tempDir = await mkdtemp(path.join(os.tmpdir(), "ra-cargo-path-"));
            try {
                const cargoBinary = path.join(
                    tempDir,
                    process.platform === "win32" ? "cargo.exe" : "cargo",
                );
                await writeFile(cargoBinary, "");

                assert.strictEqual(await cargoPath({ PATH: tempDir }), "cargo");
            } finally {
                await rm(tempDir, { recursive: true, force: true });
            }
        });

        suite.addTest("resolves cargo from provided CARGO_HOME", async () => {
            const cargoHome = await mkdtemp(path.join(os.tmpdir(), "ra-cargo-home-"));
            try {
                const binDir = path.join(cargoHome, "bin");
                await mkdir(binDir);
                const cargoBinary = path.join(
                    binDir,
                    process.platform === "win32" ? "cargo.exe" : "cargo",
                );
                await writeFile(cargoBinary, "");

                assert.strictEqual(
                    await cargoPath({ PATH: "", CARGO_HOME: cargoHome }),
                    cargoBinary,
                );
            } finally {
                await rm(cargoHome, { recursive: true, force: true });
            }
        });
    });
}
