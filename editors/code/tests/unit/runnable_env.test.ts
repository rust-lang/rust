import * as assert from 'assert';
import { prepareEnv } from '../../src/run';
import { RunnableEnvCfg } from '../../src/config';
import * as ra from '../../src/lsp_ext';

function makeRunnable(label: string): ra.Runnable {
    return {
        label,
        kind: "cargo",
        args: {
            cargoArgs: [],
            executableArgs: []
        }
    };
}

function fakePrepareEnv(runnableName: string, config: RunnableEnvCfg): Record<string, string> {
    const runnable = makeRunnable(runnableName);
    return prepareEnv(runnable, config);
}

suite('Runnable env', () => {
    test('Global config works', () => {
        const binEnv = fakePrepareEnv("run project_name", { "GLOBAL": "g" });
        assert.equal(binEnv["GLOBAL"], "g");

        const testEnv = fakePrepareEnv("test some::mod::test_name", { "GLOBAL": "g" });
        assert.equal(testEnv["GLOBAL"], "g");
    });

    test('null mask works', () => {
        const config = [
            {
                env: { DATA: "data" }
            }
        ];
        const binEnv = fakePrepareEnv("run project_name", config);
        assert.equal(binEnv["DATA"], "data");

        const testEnv = fakePrepareEnv("test some::mod::test_name", config);
        assert.equal(testEnv["DATA"], "data");
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
        const binEnv = fakePrepareEnv("run project_name", config);
        assert.equal(binEnv["DATA"], "newdata");

        const testEnv = fakePrepareEnv("test some::mod::test_name", config);
        assert.equal(testEnv["DATA"], "newdata");
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
        const binEnv = fakePrepareEnv("run project_name", config);
        assert.equal(binEnv["DATA"], "rundata");

        const testEnv = fakePrepareEnv("test some::mod::test_name", config);
        assert.equal(testEnv["DATA"], "data");

        const specialTestEnv = fakePrepareEnv("test some::mod::special_test", config);
        assert.equal(specialTestEnv["DATA"], "special_test");
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
        const testEnv = fakePrepareEnv("test some::mod::test_name", config);
        assert.equal(testEnv["DATA"], "test special");

        const specialTestEnv = fakePrepareEnv("test some::mod::another_test", config);
        assert.equal(specialTestEnv["DATA"], "data");
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
        const testEnv = fakePrepareEnv("test some::mod::test_name", config);
        assert.equal(testEnv["DATA"], "mod special");

        const specialTestEnv = fakePrepareEnv("test some::mod::another_test", config);
        assert.equal(specialTestEnv["DATA"], "mod special");
    });

});
