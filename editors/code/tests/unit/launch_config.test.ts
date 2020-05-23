import * as assert from 'assert';
import * as cargo from '../../src/cargo';

suite('Launch configuration', () => {

    suite('Lens', () => {
        test('A binary', async () => {
            const args = cargo.artifactSpec(["build", "--package", "pkg_name", "--bin", "pkg_name"]);

            assert.deepEqual(args.cargoArgs, ["build", "--package", "pkg_name", "--bin", "pkg_name", "--message-format=json"]);
            assert.deepEqual(args.filter, undefined);
        });

        test('One of Multiple Binaries', async () => {
            const args = cargo.artifactSpec(["build", "--package", "pkg_name", "--bin", "bin1"]);

            assert.deepEqual(args.cargoArgs, ["build", "--package", "pkg_name", "--bin", "bin1", "--message-format=json"]);
            assert.deepEqual(args.filter, undefined);
        });

        test('A test', async () => {
            const args = cargo.artifactSpec(["test", "--package", "pkg_name", "--lib", "--no-run"]);

            assert.deepEqual(args.cargoArgs, ["test", "--package", "pkg_name", "--lib", "--no-run", "--message-format=json"]);
            assert.notDeepEqual(args.filter, undefined);
        });
    });

    suite('QuickPick', () => {
        test('A binary', async () => {
            const args = cargo.artifactSpec(["run", "--package", "pkg_name", "--bin", "pkg_name"]);

            assert.deepEqual(args.cargoArgs, ["build", "--package", "pkg_name", "--bin", "pkg_name", "--message-format=json"]);
            assert.deepEqual(args.filter, undefined);
        });


        test('One of Multiple Binaries', async () => {
            const args = cargo.artifactSpec(["run", "--package", "pkg_name", "--bin", "bin2"]);

            assert.deepEqual(args.cargoArgs, ["build", "--package", "pkg_name", "--bin", "bin2", "--message-format=json"]);
            assert.deepEqual(args.filter, undefined);
        });

        test('A test', async () => {
            const args = cargo.artifactSpec(["test", "--package", "pkg_name", "--lib"]);

            assert.deepEqual(args.cargoArgs, ["test", "--package", "pkg_name", "--lib", "--message-format=json", "--no-run"]);
            assert.notDeepEqual(args.filter, undefined);
        });
    });
});
