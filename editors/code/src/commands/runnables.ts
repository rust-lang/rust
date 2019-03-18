import { exec } from 'child_process';
import * as util from 'util';
import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import { Server } from '../server';

interface RunnablesParams {
    textDocument: lc.TextDocumentIdentifier;
    position?: lc.Position;
}

interface Runnable {
    label: string;
    bin: string;
    args: string[];
    env: { [index: string]: string };
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
        env: spec.env
    };

    const execOption: vscode.ShellExecutionOptions = {
        cwd: '.',
        env: definition.env
    };
    const exec = new vscode.ShellExecution(
        definition.command,
        definition.args,
        execOption
    );

    const f = vscode.workspace.workspaceFolders![0];
    const t = new vscode.Task(
        definition,
        f,
        definition.label,
        TASK_SOURCE,
        exec,
        ['$rustc']
    );
    t.presentationOptions.clear = true;
    return t;
}

let prevRunnable: RunnableQuickPick | undefined;
export async function handle() {
    const editor = vscode.window.activeTextEditor;
    if (editor == null || editor.document.languageId !== 'rust') {
        return;
    }
    const textDocument: lc.TextDocumentIdentifier = {
        uri: editor.document.uri.toString()
    };
    const params: RunnablesParams = {
        textDocument,
        position: Server.client.code2ProtocolConverter.asPosition(
            editor.selection.active
        )
    };
    const runnables = await Server.client.sendRequest<Runnable[]>(
        'rust-analyzer/runnables',
        params
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
    if (item) {
        item.detail = 'rerun';
        prevRunnable = item;
        const task = createTask(item.runnable);
        return await vscode.tasks.executeTask(task);
    }
}

export async function handleSingle(runnable: Runnable) {
    const editor = vscode.window.activeTextEditor;
    if (editor == null || editor.document.languageId !== 'rust') {
        return;
    }

    const task = createTask(runnable);
    task.group = vscode.TaskGroup.Build;
    task.presentationOptions = {
        reveal: vscode.TaskRevealKind.Always,
        panel: vscode.TaskPanelKind.Dedicated,
        clear: true
    };

    return vscode.tasks.executeTask(task);
}

export const autoCargoWatchTask: vscode.Task = {
    name: 'cargo watch',
    source: 'rust-analyzer',
    definition: {
        type: 'watch'
    },
    execution: new vscode.ShellExecution('cargo', ['watch'], { cwd: '.' }),

    isBackground: true,
    problemMatchers: ['$rustc-watch'],
    presentationOptions: {
        clear: true
    },
    // Not yet exposed in the vscode.d.ts
    // https://github.com/Microsoft/vscode/blob/ea7c31d770e04b51d586b0d3944f3a7feb03afb9/src/vs/workbench/contrib/tasks/common/tasks.ts#L444-L456
    runOptions: ({
        runOn: 2 // RunOnOptions.folderOpen
    } as unknown) as vscode.RunOptions
};

/**
 * Interactively asks the user whether we should run `cargo check` in order to
 * provide inline diagnostics; the user is met with a series of dialog boxes
 * that, when accepted, allow us to `cargo install cargo-watch` and then run it.
 */
export async function interactivelyStartCargoWatch() {
    const execAsync = util.promisify(exec);

    const watch = await vscode.window.showInformationMessage(
        'Start watching changes with cargo? (Executes `cargo watch`, provides inline diagnostics)',
        'yes',
        'no'
    );
    if (watch === 'no') {
        return;
    }

    const { stderr } = await execAsync('cargo watch --version').catch(e => e);
    if (stderr.includes('no such subcommand: `watch`')) {
        const msg =
            'The `cargo-watch` subcommand is not installed. Install? (takes ~1-2 minutes)';
        const install = await vscode.window.showInformationMessage(
            msg,
            'yes',
            'no'
        );
        if (install === 'no') {
            return;
        }

        const label = 'install-cargo-watch';
        const taskFinished = new Promise((resolve, reject) => {
            let disposable = vscode.tasks.onDidEndTask(({ execution }) => {
                if (execution.task.name === label) {
                    disposable.dispose();
                    resolve();
                }
            });
        });

        vscode.tasks.executeTask(
            createTask({
                label,
                bin: 'cargo',
                args: ['install', 'cargo-watch'],
                env: {}
            })
        );
        await taskFinished;
        const { stderr } = await execAsync('cargo watch --version').catch(
            e => e
        );
        if (stderr !== '') {
            vscode.window.showErrorMessage(
                `Couldn't install \`cargo-\`watch: ${stderr}`
            );
            return;
        }
    }

    vscode.tasks.executeTask(autoCargoWatchTask);
}
