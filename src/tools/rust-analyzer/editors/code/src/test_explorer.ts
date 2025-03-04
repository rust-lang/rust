import * as vscode from "vscode";
import type * as lc from "vscode-languageclient/node";
import * as ra from "./lsp_ext";

import type { Ctx } from "./ctx";
import { startDebugSession } from "./debug";

export const prepareTestExplorer = (
    ctx: Ctx,
    testController: vscode.TestController,
    client: lc.LanguageClient,
) => {
    let currentTestRun: vscode.TestRun | undefined;
    let idToTestMap: Map<string, vscode.TestItem> = new Map();
    const fileToTestMap: Map<string, vscode.TestItem[]> = new Map();
    const idToRunnableMap: Map<string, ra.Runnable> = new Map();

    testController.createRunProfile(
        "Run Tests",
        vscode.TestRunProfileKind.Run,
        async (request: vscode.TestRunRequest, cancelToken: vscode.CancellationToken) => {
            if (currentTestRun) {
                await client.sendNotification(ra.abortRunTest);
                while (currentTestRun) {
                    await new Promise((resolve) => setTimeout(resolve, 1));
                }
            }

            currentTestRun = testController.createTestRun(request);
            cancelToken.onCancellationRequested(async () => {
                await client.sendNotification(ra.abortRunTest);
            });
            const include = request.include?.map((x) => x.id);
            const exclude = request.exclude?.map((x) => x.id);
            await client.sendRequest(ra.runTest, { include, exclude });
        },
        true,
        undefined,
        false,
    );

    testController.createRunProfile(
        "Debug Tests",
        vscode.TestRunProfileKind.Debug,
        async (request: vscode.TestRunRequest) => {
            if (request.include?.length !== 1 || request.exclude?.length !== 0) {
                await vscode.window.showErrorMessage("You can debug only one test at a time");
                return;
            }
            const id = request.include[0]!.id;
            const runnable = idToRunnableMap.get(id);
            if (!runnable) {
                await vscode.window.showErrorMessage("You can debug only one test at a time");
                return;
            }
            await startDebugSession(ctx, runnable);
        },
        true,
        undefined,
        false,
    );

    const deleteTest = (item: vscode.TestItem, parentList: vscode.TestItemCollection) => {
        parentList.delete(item.id);
        idToTestMap.delete(item.id);
        idToRunnableMap.delete(item.id);
        if (item.uri) {
            fileToTestMap.set(
                item.uri.toString(),
                fileToTestMap.get(item.uri.toString())!.filter((t) => t.id !== item.id),
            );
        }
    };

    const addTest = (item: ra.TestItem) => {
        const parentList = item.parent
            ? idToTestMap.get(item.parent)!.children
            : testController.items;
        const oldTest = parentList.get(item.id);
        const uri = item.textDocument?.uri ? vscode.Uri.parse(item.textDocument?.uri) : undefined;
        const range =
            item.range &&
            new vscode.Range(
                new vscode.Position(item.range.start.line, item.range.start.character),
                new vscode.Position(item.range.end.line, item.range.end.character),
            );
        if (oldTest) {
            if (oldTest.uri?.toString() === uri?.toString()) {
                oldTest.range = range;
                return;
            }
            deleteTest(oldTest, parentList);
        }
        const iconToVscodeMap = {
            package: "package",
            module: "symbol-module",
            test: "beaker",
        };
        const test = testController.createTestItem(
            item.id,
            `$(${iconToVscodeMap[item.kind]}) ${item.label}`,
            uri,
        );
        test.range = range;
        test.canResolveChildren = item.canResolveChildren;
        idToTestMap.set(item.id, test);
        if (uri) {
            if (!fileToTestMap.has(uri.toString())) {
                fileToTestMap.set(uri.toString(), []);
            }
            fileToTestMap.get(uri.toString())!.push(test);
        }
        if (item.runnable) {
            idToRunnableMap.set(item.id, item.runnable);
        }
        parentList.add(test);
    };

    const addTestGroup = (testsAndScope: ra.DiscoverTestResults) => {
        const { tests, scope, scopeFile } = testsAndScope;
        const testSet: Set<string> = new Set();
        for (const test of tests) {
            addTest(test);
            testSet.add(test.id);
        }
        // FIXME(hack_recover_crate_name): We eagerly resolve every test if we got a lazy top level response (detected
        // by checking that `scope` is empty). This is not a good thing and wastes cpu and memory unnecessarily, so we
        // should remove it.
        if (!scope && !scopeFile) {
            for (const test of tests) {
                void testController.resolveHandler!(idToTestMap.get(test.id));
            }
        }
        if (scope) {
            const recursivelyRemove = (tests: vscode.TestItemCollection) => {
                for (const [, test] of tests) {
                    if (!testSet.has(test.id)) {
                        deleteTest(test, tests);
                    } else {
                        recursivelyRemove(test.children);
                    }
                }
            };
            for (const root of scope) {
                recursivelyRemove(idToTestMap.get(root)!.children);
            }
        }
        if (scopeFile) {
            const removeByFile = (file: vscode.Uri) => {
                const testsToBeRemoved = (fileToTestMap.get(file.toString()) || []).filter(
                    (t) => !testSet.has(t.id),
                );
                for (const test of testsToBeRemoved) {
                    const parentList = test.parent?.children || testController.items;
                    deleteTest(test, parentList);
                }
            };
            for (const file of scopeFile) {
                removeByFile(vscode.Uri.parse(file.uri));
            }
        }
    };

    ctx.pushClientCleanup(
        client.onNotification(ra.discoveredTests, (results) => {
            addTestGroup(results);
        }),
    );

    ctx.pushClientCleanup(
        client.onNotification(ra.endRunTest, () => {
            currentTestRun!.end();
            currentTestRun = undefined;
        }),
    );

    ctx.pushClientCleanup(
        client.onNotification(ra.appendOutputToRunTest, (output) => {
            currentTestRun!.appendOutput(`${output}\r\n`);
        }),
    );

    ctx.pushClientCleanup(
        client.onNotification(ra.changeTestState, (results) => {
            const test = idToTestMap.get(results.testId)!;
            if (results.state.tag === "failed") {
                currentTestRun!.failed(test, new vscode.TestMessage(results.state.message));
            } else if (results.state.tag === "passed") {
                currentTestRun!.passed(test);
            } else if (results.state.tag === "started") {
                currentTestRun!.started(test);
            } else if (results.state.tag === "skipped") {
                currentTestRun!.skipped(test);
            } else if (results.state.tag === "enqueued") {
                currentTestRun!.enqueued(test);
            }
        }),
    );

    testController.resolveHandler = async (item) => {
        const results = await client.sendRequest(ra.discoverTest, { testId: item?.id });
        addTestGroup(results);
    };

    testController.refreshHandler = async () => {
        testController.items.forEach((t) => {
            testController.items.delete(t.id);
        });
        idToTestMap = new Map();
        await testController.resolveHandler!(undefined);
    };
};
