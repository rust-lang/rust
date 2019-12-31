import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import { Ctx, Cmd } from '../ctx';

export function run(ctx: Ctx): Cmd {
    let prevRunnable: RunnableQuickPick | undefined;

    return async () => {
        const editor = ctx.activeRustEditor;
        const client = ctx.client;
        if (!editor || !client) return;

        const textDocument: lc.TextDocumentIdentifier = {
            uri: editor.document.uri.toString(),
        };
        const params: RunnablesParams = {
            textDocument,
            position: client.code2ProtocolConverter.asPosition(
                editor.selection.active,
            ),
        };
        const runnables = await client.sendRequest<Runnable[]>(
            'rust-analyzer/runnables',
            params,
        );
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
            items.push(new RunnableQuickPick(r));
        }
        const item = await vscode.window.showQuickPick(items);
        if (!item) return;

        item.detail = 'rerun';
        prevRunnable = item;
        const task = createTask(item.runnable);
        return await vscode.tasks.executeTask(task);
    };
}

export function runSingle(ctx: Ctx): Cmd {
    return async (runnable: Runnable) => {
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

interface RunnablesParams {
    textDocument: lc.TextDocumentIdentifier;
    position?: lc.Position;
}

interface Runnable {
    label: string;
    bin: string;
    args: string[];
    env: { [index: string]: string };
    cwd?: string;
}

class RunnableQuickPick implements vscode.QuickPickItem {
    public label: string;
    public description?: string | undefined;
    public detail?: string | undefined;
    public picked?: boolean | undefined;

    constructor(public runnable: Runnable) {
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

function createTask(spec: Runnable): vscode.Task {
    const TASK_SOURCE = 'Rust';
    const definition: CargoTaskDefinition = {
        type: 'cargo',
        label: spec.label,
        command: spec.bin,
        args: spec.args,
        env: spec.env,
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
