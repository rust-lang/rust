import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';
import * as ra from './lsp_ext';

import { Ctx, Cmd } from './ctx';
import { startDebugSession, getDebugConfiguration } from './debug';

const quickPickButtons = [{ iconPath: new vscode.ThemeIcon("save"), tooltip: "Save as a launch.json configurtation." }];

export async function selectRunnable(ctx: Ctx, prevRunnable?: RunnableQuickPick, debuggeeOnly = false, showButtons: boolean = true): Promise<RunnableQuickPick | undefined> {
    const editor = ctx.activeRustEditor;
    const client = ctx.client;
    if (!editor || !client) return;

    const textDocument: lc.TextDocumentIdentifier = {
        uri: editor.document.uri.toString(),
    };

    const runnables = await client.sendRequest(ra.runnables, {
        textDocument,
        position: client.code2ProtocolConverter.asPosition(
            editor.selection.active,
        ),
    });
    const items: RunnableQuickPick[] = [];
    if (prevRunnable) {
        items.push(prevRunnable);
    }
    for (const r of runnables) {
        if (
            prevRunnable &&
            JSON.stringify(prevRunnable.runnable) === JSON.stringify(r)
        ) {
            continue;
        }

        if (debuggeeOnly && (r.label.startsWith('doctest') || r.label.startsWith('cargo'))) {
            continue;
        }
        items.push(new RunnableQuickPick(r));
    }

    if (items.length === 0) {
        // it is the debug case, run always has at least 'cargo check ...'
        // see crates\rust-analyzer\src\main_loop\handlers.rs, handle_runnables
        vscode.window.showErrorMessage("There's no debug target!");
        return;
    }

    return await new Promise((resolve) => {
        const disposables: vscode.Disposable[] = [];
        const close = (result?: RunnableQuickPick) => {
            resolve(result);
            disposables.forEach(d => d.dispose());
        };

        const quickPick = vscode.window.createQuickPick<RunnableQuickPick>();
        quickPick.items = items;
        quickPick.title = "Select Runnable";
        if (showButtons) {
            quickPick.buttons = quickPickButtons;
        }
        disposables.push(
            quickPick.onDidHide(() => close()),
            quickPick.onDidAccept(() => close(quickPick.selectedItems[0])),
            quickPick.onDidTriggerButton((_button) => {
                (async () => await makeDebugConfig(ctx, quickPick.activeItems[0]))();
                close();
            }),
            quickPick.onDidChangeActive((active) => {
                if (showButtons && active.length > 0) {
                    if (active[0].label.startsWith('cargo')) {
                        // save button makes no sense for `cargo test` or `cargo check`
                        quickPick.buttons = [];
                    } else if (quickPick.buttons.length === 0) {
                        quickPick.buttons = quickPickButtons;
                    }
                }
            }),
            quickPick
        );
        quickPick.show();
    });
}

export function runSingle(ctx: Ctx): Cmd {
    return async (runnable: ra.Runnable) => {
        const editor = ctx.activeRustEditor;
        if (!editor) return;

        const task = createTask(runnable);
        task.group = vscode.TaskGroup.Build;
        task.presentationOptions = {
            reveal: vscode.TaskRevealKind.Always,
            panel: vscode.TaskPanelKind.Dedicated,
            clear: true,
        };

        return vscode.tasks.executeTask(task);
    };
}

export function debug(ctx: Ctx): Cmd {
    let prevDebuggee: RunnableQuickPick | undefined;

    return async () => {
        const item = await selectRunnable(ctx, prevDebuggee, true);
        if (!item) return;

        item.detail = 'restart';
        prevDebuggee = item;
        return await startDebugSession(ctx, item.runnable);
    };
}

export function debugSingle(ctx: Ctx): Cmd {
    return async (config: ra.Runnable) => {
        await startDebugSession(ctx, config);
    };
}

async function makeDebugConfig(ctx: Ctx, item: RunnableQuickPick): Promise<void> {
    const scope = ctx.activeRustEditor?.document.uri;
    if (!scope) return;

    const debugConfig = await getDebugConfiguration(ctx, item.runnable);
    if (!debugConfig) return;

    const wsLaunchSection = vscode.workspace.getConfiguration("launch", scope);
    const configurations = wsLaunchSection.get<any[]>("configurations") || [];

    const index = configurations.findIndex(c => c.name === debugConfig.name);
    if (index !== -1) {
        const answer = await vscode.window.showErrorMessage(`Launch configuration '${debugConfig.name}' already exists!`, 'Cancel', 'Update');
        if (answer === "Cancel") return;

        configurations[index] = debugConfig;
    } else {
        configurations.push(debugConfig);
    }

    await wsLaunchSection.update("configurations", configurations);
}

export function newDebugConfig(ctx: Ctx): Cmd {
    return async () => {
        const item = await selectRunnable(ctx, undefined, true, false);
        if (!item) return;

        await makeDebugConfig(ctx, item);
    };
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

interface CargoTaskDefinition extends vscode.TaskDefinition {
    type: 'cargo';
    label: string;
    command: string;
    args: string[];
    env?: { [key: string]: string };
}

export function createTask(spec: ra.Runnable): vscode.Task {
    const TASK_SOURCE = 'Rust';
    const definition: CargoTaskDefinition = {
        type: 'cargo',
        label: spec.label,
        command: spec.bin,
        args: spec.extraArgs ? [...spec.args, '--', ...spec.extraArgs] : spec.args,
        env: Object.assign({}, process.env, spec.env),
    };

    const execOption: vscode.ShellExecutionOptions = {
        cwd: spec.cwd || '.',
        env: definition.env,
    };
    const exec = new vscode.ShellExecution(
        definition.command,
        definition.args,
        execOption,
    );

    const f = vscode.workspace.workspaceFolders![0];
    const t = new vscode.Task(
        definition,
        f,
        definition.label,
        TASK_SOURCE,
        exec,
        ['$rustc'],
    );
    t.presentationOptions.clear = true;
    return t;
}
