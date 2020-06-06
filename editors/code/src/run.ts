import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';
import * as ra from './lsp_ext';
import * as toolchain from "./toolchain";

import { Ctx } from './ctx';
import { makeDebugConfig } from './debug';

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
                (async () => await makeDebugConfig(ctx, quickPick.activeItems[0].runnable))();
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

export function createTask(runnable: ra.Runnable): vscode.Task {
    const TASK_SOURCE = 'Rust';

    let command;
    switch (runnable.kind) {
        case "cargo": command = toolchain.getPathForExecutable("cargo");
    }
    const args = [...runnable.args.cargoArgs]; // should be a copy!
    if (runnable.args.executableArgs.length > 0) {
        args.push('--', ...runnable.args.executableArgs);
    }
    const definition: CargoTaskDefinition = {
        type: 'cargo',
        label: runnable.label,
        command,
        args,
        env: Object.assign({}, process.env as { [key: string]: string }, { "RUST_BACKTRACE": "short" }),
    };

    const execOption: vscode.ShellExecutionOptions = {
        cwd: runnable.args.workspaceRoot || '.',
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
