import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';
import * as ra from '../rust-analyzer-api';

import { Ctx, Cmd } from '../ctx';
import { startDebugSession, getDebugConfiguration } from '../debug';

async function selectRunnable(ctx: Ctx, prevRunnable: RunnableQuickPick | undefined): Promise<RunnableQuickPick | undefined> {
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
        items.push(new RunnableQuickPick(r));
    }
    return await vscode.window.showQuickPick(items);
}

export function run(ctx: Ctx): Cmd {
    let prevRunnable: RunnableQuickPick | undefined;

    return async () => {
        const item = await selectRunnable(ctx, prevRunnable);
        if (!item) return;

        item.detail = 'rerun';
        prevRunnable = item;
        const task = createTask(item.runnable);
        return await vscode.tasks.executeTask(task);
    };
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
        const item = await selectRunnable(ctx, prevDebuggee);
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

export function newDebugConfig(ctx: Ctx): Cmd {
    return async () => {
        const scope = ctx.activeRustEditor?.document.uri;
        if (!scope) return;

        const item = await selectRunnable(ctx, undefined);
        if (!item) return;

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
    };
}

class RunnableQuickPick implements vscode.QuickPickItem {
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

function createTask(spec: ra.Runnable): vscode.Task {
    const TASK_SOURCE = 'Rust';
    const definition: CargoTaskDefinition = {
        type: 'cargo',
        label: spec.label,
        command: spec.bin,
        args: spec.extraArgs ? [...spec.args, '--', ...spec.extraArgs] : spec.args,
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
