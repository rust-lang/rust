import * as assert from 'assert';
import { Cargo } from '../../src/toolchain';

suite('Launch configuration', () => {

    suite('Lens', () => {
        test('A binary', async () => {
            const args = Cargo.artifactSpec(["build", "--package", "pkg_name", "--bin", "pkg_name"]);

            assert.deepStrictEqual(args.cargoArgs, ["build", "--package", "pkg_name", "--bin", "pkg_name", "--message-format=json"]);
            assert.deepStrictEqual(args.filter, undefined);
        });

        test('One of Multiple Binaries', async () => {
            const args = Cargo.artifactSpec(["build", "--package", "pkg_name", "--bin", "bin1"]);

            assert.deepStrictEqual(args.cargoArgs, ["build", "--package", "pkg_name", "--bin", "bin1", "--message-format=json"]);
            assert.deepStrictEqual(args.filter, undefined);
        });

        test('A test', async () => {
            const args = Cargo.artifactSpec(["test", "--package", "pkg_name", "--lib", "--no-run"]);

            assert.deepStrictEqual(args.cargoArgs, ["test", "--package", "pkg_name", "--lib", "--no-run", "--message-format=json"]);
            assert.notDeepStrictEqual(args.filter, undefined);
        });
    });

    suite('QuickPick', () => {
        test('A binary', async () => {
            const args = Cargo.artifactSpec(["run", "--package", "pkg_name", "--bin", "pkg_name"]);

            assert.deepStrictEqual(args.cargoArgs, ["build", "--package", "pkg_name", "--bin", "pkg_name", "--message-format=json"]);
            assert.deepStrictEqual(args.filter, undefined);
        });


        test('One of Multiple Binaries', async () => {
            const args = Cargo.artifactSpec(["run", "--package", "pkg_name", "--bin", "bin2"]);

            assert.deepStrictEqual(args.cargoArgs, ["build", "--package", "pkg_name", "--bin", "bin2", "--message-format=json"]);
            assert.deepStrictEqual(args.filter, undefined);
        });

        test('A test', async () => {
            const args = Cargo.artifactSpec(["test", "--package", "pkg_name", "--lib"]);

            assert.deepStrictEqual(args.cargoArgs, ["test", "--package", "pkg_name", "--lib", "--message-format=json", "--no-run"]);
            assert.notDeepStrictEqual(args.filter, undefined);
        });
    });
});
