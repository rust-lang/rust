import * as vscode from "vscode";
import type * as lc from "vscode-languageclient";
import * as ra from "./lsp_ext";
import * as tasks from "./tasks";

import type { CtxInit } from "./ctx";
import { makeDebugConfig } from "./debug";
import type { Config, RunnableEnvCfg, RunnableEnvCfgItem } from "./config";
import type { LanguageClient } from "vscode-languageclient/node";
import { unwrapUndefinable, type RustEditor } from "./util";

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
        // see crates\rust-analyzer\src\handlers\request.rs, handle_runnables
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

export function prepareBaseEnv(
    inheritEnv: boolean,
    base?: Record<string, string>,
): Record<string, string> {
    const env: Record<string, string> = { RUST_BACKTRACE: "short" };
    if (inheritEnv) {
        Object.assign(env, process.env);
    }
    if (base) {
        Object.assign(env, base);
    }
    return env;
}

export function prepareEnv(
    inheritEnv: boolean,
    label: string,
    runnableArgs: ra.CargoRunnableArgs,
    runnableEnvCfg?: RunnableEnvCfg,
): Record<string, string> {
    const env = prepareBaseEnv(inheritEnv, runnableArgs.environment);
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
                const masked = !it.mask || new RegExp(it.mask).test(label);
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

export async function createTaskFromRunnable(
    runnable: ra.Runnable,
    config: Config,
): Promise<vscode.Task> {
    const target = vscode.workspace.workspaceFolders?.[0];

    let definition: tasks.TaskDefinition;
    let options;
    let cargo = "cargo";
    if (runnable.kind === "cargo") {
        const runnableArgs = runnable.args;
        let args = createCargoArgs(runnableArgs);

        if (runnableArgs.overrideCargo) {
            // Split on spaces to allow overrides like "wrapper cargo".
            const cargoParts = runnableArgs.overrideCargo.split(" ");

            cargo = unwrapUndefinable(cargoParts[0]);
            args = [...cargoParts.slice(1), ...args];
        }

        definition = {
            type: tasks.CARGO_TASK_TYPE,
            command: unwrapUndefinable(args[0]),
            args: args.slice(1),
        };
        options = {
            cwd: runnableArgs.workspaceRoot || ".",
            env: prepareEnv(true, runnable.label, runnableArgs, config.runnablesExtraEnv),
        };
    } else {
        const runnableArgs = runnable.args;
        definition = {
            type: tasks.SHELL_TASK_TYPE,
            command: runnableArgs.program,
            args: runnableArgs.args,
        };
        options = {
            cwd: runnableArgs.cwd,
            env: prepareBaseEnv(true),
        };
    }

    const exec = await tasks.targetToExecution(definition, options, cargo);
    const task = await tasks.buildRustTask(
        target,
        definition,
        runnable.label,
        config.problemMatcher,
        exec,
    );

    task.presentationOptions.clear = true;
    // Sadly, this doesn't prevent focus stealing if the terminal is currently
    // hidden, and will become revealed due to task execution.
    task.presentationOptions.focus = false;

    return task;
}

export function createCargoArgs(runnableArgs: ra.CargoRunnableArgs): string[] {
    const args = [...runnableArgs.cargoArgs]; // should be a copy!
    if (runnableArgs.executableArgs.length > 0) {
        args.push("--", ...runnableArgs.executableArgs);
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

        if (debuggeeOnly && r.label.startsWith("doctest")) {
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
