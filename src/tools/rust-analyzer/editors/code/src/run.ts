import * as vscode from "vscode";
import type * as lc from "vscode-languageclient";
import * as ra from "./lsp_ext";
import * as tasks from "./tasks";
import * as toolchain from "./toolchain";

import type { CtxInit } from "./ctx";
import { makeDebugConfig } from "./debug";
import type { Config, RunnableEnvCfg, RunnableEnvCfgItem } from "./config";
import { unwrapUndefinable } from "./undefinable";
import type { LanguageClient } from "vscode-languageclient/node";
import type { RustEditor } from "./util";

const quickPickButtons = [
    { iconPath: new vscode.ThemeIcon("save"), tooltip: "Save as a launch.json configuration." },
];

export async function selectRunnable(
    ctx: CtxInit,
    prevRunnable?: RunnableQuickPick,
    debuggeeOnly = false,
    showButtons: boolean = true,
): Promise<RunnableQuickPick | undefined> {
    const editor = ctx.activeRustEditor;
    if (!editor) return;

    // show a placeholder while we get the runnables from the server
    const quickPick = vscode.window.createQuickPick();
    quickPick.title = "Select Runnable";
    if (showButtons) {
        quickPick.buttons = quickPickButtons;
    }
    quickPick.items = [{ label: "Looking for runnables..." }];
    quickPick.activeItems = [];
    quickPick.show();

    const runnables = await getRunnables(ctx.client, editor, prevRunnable, debuggeeOnly);

    if (runnables.length === 0) {
        // it is the debug case, run always has at least 'cargo check ...'
        // see crates\rust-analyzer\src\main_loop\handlers.rs, handle_runnables
        await vscode.window.showErrorMessage("There's no debug target!");
        quickPick.dispose();
        return;
    }

    // clear the list before we hook up listeners to avoid invoking them
    // if the user happens to accept the placeholder item
    quickPick.items = [];

    return await populateAndGetSelection(
        quickPick as vscode.QuickPick<RunnableQuickPick>,
        runnables,
        ctx,
        showButtons,
    );
}

export class RunnableQuickPick implements vscode.QuickPickItem {
    public label: string;
    public description?: string | undefined;
    public detail?: string | undefined;
    public picked?: boolean | undefined;

    constructor(public runnable: ra.Runnable) {
        this.label = runnable.label;
    }
}

export function prepareEnv(
    runnable: ra.Runnable,
    runnableEnvCfg: RunnableEnvCfg,
): Record<string, string> {
    const env: Record<string, string> = { RUST_BACKTRACE: "short" };

    if (runnable.args.expectTest) {
        env["UPDATE_EXPECT"] = "1";
    }

    Object.assign(env, process.env as { [key: string]: string });
    const platform = process.platform;

    const checkPlatform = (it: RunnableEnvCfgItem) => {
        if (it.platform) {
            const platforms = Array.isArray(it.platform) ? it.platform : [it.platform];
            return platforms.indexOf(platform) >= 0;
        }
        return true;
    };

    if (runnableEnvCfg) {
        if (Array.isArray(runnableEnvCfg)) {
            for (const it of runnableEnvCfg) {
                const masked = !it.mask || new RegExp(it.mask).test(runnable.label);
                if (masked && checkPlatform(it)) {
                    Object.assign(env, it.env);
                }
            }
        } else {
            Object.assign(env, runnableEnvCfg);
        }
    }

    return env;
}

export async function createTask(runnable: ra.Runnable, config: Config): Promise<vscode.Task> {
    if (runnable.kind !== "cargo") {
        // rust-analyzer supports only one kind, "cargo"
        // do not use tasks.TASK_TYPE here, these are completely different meanings.

        throw `Unexpected runnable kind: ${runnable.kind}`;
    }

    let program: string;
    let args = createArgs(runnable);
    if (runnable.args.overrideCargo) {
        // Split on spaces to allow overrides like "wrapper cargo".
        const cargoParts = runnable.args.overrideCargo.split(" ");

        program = unwrapUndefinable(cargoParts[0]);
        args = [...cargoParts.slice(1), ...args];
    } else {
        program = await toolchain.cargoPath();
    }

    const definition: tasks.RustTargetDefinition = {
        type: tasks.TASK_TYPE,
        program,
        args,
        cwd: runnable.args.workspaceRoot || ".",
        env: prepareEnv(runnable, config.runnablesExtraEnv),
        overrideCargo: runnable.args.overrideCargo,
    };

    // eslint-disable-next-line @typescript-eslint/no-unnecessary-type-assertion
    const target = vscode.workspace.workspaceFolders![0]; // safe, see main activate()
    const task = await tasks.buildRustTask(
        target,
        definition,
        runnable.label,
        config.problemMatcher,
        config.cargoRunner,
        true,
    );

    task.presentationOptions.clear = true;
    // Sadly, this doesn't prevent focus stealing if the terminal is currently
    // hidden, and will become revealed due to task execution.
    task.presentationOptions.focus = false;

    return task;
}

export function createArgs(runnable: ra.Runnable): string[] {
    const args = [...runnable.args.cargoArgs]; // should be a copy!
    if (runnable.args.cargoExtraArgs) {
        args.push(...runnable.args.cargoExtraArgs); // Append user-specified cargo options.
    }
    if (runnable.args.executableArgs.length > 0) {
        args.push("--", ...runnable.args.executableArgs);
    }
    return args;
}

async function getRunnables(
    client: LanguageClient,
    editor: RustEditor,
    prevRunnable?: RunnableQuickPick,
    debuggeeOnly = false,
): Promise<RunnableQuickPick[]> {
    const textDocument: lc.TextDocumentIdentifier = {
        uri: editor.document.uri.toString(),
    };

    const runnables = await client.sendRequest(ra.runnables, {
        textDocument,
        position: client.code2ProtocolConverter.asPosition(editor.selection.active),
    });
    const items: RunnableQuickPick[] = [];
    if (prevRunnable) {
        items.push(prevRunnable);
    }
    for (const r of runnables) {
        if (prevRunnable && JSON.stringify(prevRunnable.runnable) === JSON.stringify(r)) {
            continue;
        }

        if (debuggeeOnly && (r.label.startsWith("doctest") || r.label.startsWith("cargo"))) {
            continue;
        }
        items.push(new RunnableQuickPick(r));
    }

    return items;
}

async function populateAndGetSelection(
    quickPick: vscode.QuickPick<RunnableQuickPick>,
    runnables: RunnableQuickPick[],
    ctx: CtxInit,
    showButtons: boolean,
): Promise<RunnableQuickPick | undefined> {
    return new Promise((resolve) => {
        const disposables: vscode.Disposable[] = [];
        const close = (result?: RunnableQuickPick) => {
            resolve(result);
            disposables.forEach((d) => d.dispose());
        };
        disposables.push(
            quickPick.onDidHide(() => close()),
            quickPick.onDidAccept(() => close(quickPick.selectedItems[0] as RunnableQuickPick)),
            quickPick.onDidTriggerButton(async (_button) => {
                const runnable = unwrapUndefinable(
                    quickPick.activeItems[0] as RunnableQuickPick,
                ).runnable;
                await makeDebugConfig(ctx, runnable);
                close();
            }),
            quickPick.onDidChangeActive((activeList) => {
                if (showButtons && activeList.length > 0) {
                    const active = unwrapUndefinable(activeList[0]);
                    if (active.label.startsWith("cargo")) {
                        // save button makes no sense for `cargo test` or `cargo check`
                        quickPick.buttons = [];
                    } else if (quickPick.buttons.length === 0) {
                        quickPick.buttons = quickPickButtons;
                    }
                }
            }),
            quickPick,
        );
        // populate the list with the actual runnables
        quickPick.items = runnables;
    });
}
