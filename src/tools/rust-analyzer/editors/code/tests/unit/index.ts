import * as assert from "node:assert/strict";
import { readdir } from "fs/promises";
import * as path from "path";
import { pathToFileURL } from "url";

class Test {
    readonly name: string;
    readonly promise: Promise<void>;

    constructor(name: string, promise: Promise<void>) {
        this.name = name;
        this.promise = promise;
    }
}

class Suite {
    tests: Test[];

    constructor() {
        this.tests = [];
    }

    public addTest(name: string, f: () => Promise<void>): void {
        const test = new Test(name, f());
        this.tests.push(test);
    }

    public async run(): Promise<void> {
        let failed = 0;
        for (const test of this.tests) {
            try {
                await test.promise;
                ok(`  ✔ ${test.name}`);
            } catch (e) {
                assert.ok(e instanceof Error);
                error(`  ✖︎ ${test.name}\n  ${e.stack}`);
                failed += 1;
            }
        }
        if (failed) {
            const plural = failed > 1 ? "s" : "";
            throw new Error(`${failed} failed test${plural}`);
        }
    }
}

export class Context {
    public async suite(name: string, f: (ctx: Suite) => void): Promise<void> {
        const ctx = new Suite();
        f(ctx);
        try {
            ok(`⌛︎ ${name}`);
            await ctx.run();
            ok(`✔ ${name}`);
        } catch (e) {
            assert.ok(e instanceof Error);
            error(`✖︎ ${name}\n  ${e.stack}`);
            throw e;
        }
    }
}

export async function run(): Promise<void> {
    const context = new Context();

    const testFiles = (await readdir(path.resolve(__dirname))).filter((name) =>
        name.endsWith(".test.js"),
    );
    for (const testFile of testFiles) {
        try {
            const testModule = await import(pathToFileURL(path.resolve(__dirname, testFile)).href);
            await testModule.getTests(context);
        } catch (e) {
            error(`${e}`);
            throw e;
        }
    }
}

function ok(message: string): void {
    // eslint-disable-next-line no-console
    console.log(`\x1b[32m${message}\x1b[0m`);
}

function error(message: string): void {
    // eslint-disable-next-line no-console
    console.error(`\x1b[31m${message}\x1b[0m`);
}
