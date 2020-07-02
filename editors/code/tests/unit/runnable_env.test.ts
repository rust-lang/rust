import * as assert from 'assert';
import { prepareEnv } from '../../src/run';
import { RunnableEnvCfg } from '../../src/config';
import * as ra from '../../src/lsp_ext';

function make_runnable(label: string): ra.Runnable {
    return {
        label,
        kind: "cargo",
        args: {
            cargoArgs: [],
            executableArgs: []
        }
    }
}

function fakePrepareEnv(runnable_name: string, config: RunnableEnvCfg) : Record<string, string> {
    const runnable = make_runnable(runnable_name);
    return prepareEnv(runnable, config);
}

suite('Runnable env', () => {
    test('Global config works', () => {
        const bin_env = fakePrepareEnv("run project_name", {"GLOBAL": "g"});
        assert.equal(bin_env["GLOBAL"], "g");

        const test_env = fakePrepareEnv("test some::mod::test_name", {"GLOBAL": "g"});
        assert.equal(test_env["GLOBAL"], "g");
    });

    test('null mask works', () => {
        const config = [
            {
                env: { DATA: "data" }
            }
        ];
        const bin_env = fakePrepareEnv("run project_name", config);
        assert.equal(bin_env["DATA"], "data");

        const test_env = fakePrepareEnv("test some::mod::test_name", config);
        assert.equal(test_env["DATA"], "data");
    });

    test('order works', () => {
        const config = [
            {
                env: { DATA: "data" }
            },
            {
                env: { DATA: "newdata" }
            }
        ];
        const bin_env = fakePrepareEnv("run project_name", config);
        assert.equal(bin_env["DATA"], "newdata");

        const test_env = fakePrepareEnv("test some::mod::test_name", config);
        assert.equal(test_env["DATA"], "newdata");
    });

    test('mask works', () => {
        const config = [
            {
                env: { DATA: "data" }
            },
            {
                mask: "^run",
                env: { DATA: "rundata" }
            },
            {
                mask: "special_test$",
                env: { DATA: "special_test" }
            }
        ];
        const bin_env = fakePrepareEnv("run project_name", config);
        assert.equal(bin_env["DATA"], "rundata");

        const test_env = fakePrepareEnv("test some::mod::test_name", config);
        assert.equal(test_env["DATA"], "data");

        const special_test_env = fakePrepareEnv("test some::mod::special_test", config);
        assert.equal(special_test_env["DATA"], "special_test");
    });

    test('exact test name works', () => {
        const config = [
            {
                env: { DATA: "data" }
            },
            {
                mask: "some::mod::test_name",
                env: { DATA: "test special" }
            }
        ];
        const test_env = fakePrepareEnv("test some::mod::test_name", config);
        assert.equal(test_env["DATA"], "test special");

        const special_test_env = fakePrepareEnv("test some::mod::another_test", config);
        assert.equal(special_test_env["DATA"], "data");
    });

    test('test mod name works', () => {
        const config = [
            {
                env: { DATA: "data" }
            },
            {
                mask: "some::mod",
                env: { DATA: "mod special" }
            }
        ];
        const test_env = fakePrepareEnv("test some::mod::test_name", config);
        assert.equal(test_env["DATA"], "mod special");

        const special_test_env = fakePrepareEnv("test some::mod::another_test", config);
        assert.equal(special_test_env["DATA"], "mod special");
    });

});
