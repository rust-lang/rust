import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient'
import { Server } from '../server';

interface RunnablesParams {
    textDocument: lc.TextDocumentIdentifier,
    position?: lc.Position,
}

interface Runnable {
    range: lc.Range;
    label: string;
    bin: string;
    args: string[];
    env: { [index: string]: string },
}

class RunnableQuickPick implements vscode.QuickPickItem {
    label: string;
    description?: string | undefined;
    detail?: string | undefined;
    picked?: boolean | undefined;

    constructor(public runnable: Runnable) {
        this.label = runnable.label
    }
}

interface CargoTaskDefinition extends vscode.TaskDefinition {
    type: 'cargo';
    label: string;
    command: string;
    args: Array<string>;
    env?: { [key: string]: string };
}

function createTask(spec: Runnable): vscode.Task {
    const TASK_SOURCE = 'Rust';
    let definition: CargoTaskDefinition = {
        type: 'cargo',
        label: 'cargo',
        command: spec.bin,
        args: spec.args,
        env: spec.env
    }

    let execCmd = `${definition.command} ${definition.args.join(' ')}`;
    let execOption: vscode.ShellExecutionOptions = {
        cwd: '.',
        env: definition.env,
    };
    let exec = new vscode.ShellExecution(`clear; ${execCmd}`, execOption);

    let f = vscode.workspace.workspaceFolders![0]
    let t = new vscode.Task(definition, f, definition.label, TASK_SOURCE, exec, ['$rustc']);
    return t;
}

let prevRunnable: RunnableQuickPick | undefined = undefined
export async function handle() {
    let editor = vscode.window.activeTextEditor
    if (editor == null || editor.document.languageId != "rust") return
    let textDocument: lc.TextDocumentIdentifier = {
        uri: editor.document.uri.toString()
    }
    let params: RunnablesParams = {
        textDocument,
        position: Server.client.code2ProtocolConverter.asPosition(editor.selection.active)
    }
    let runnables = await Server.client.sendRequest<Runnable[]>('m/runnables', params)
    let items: RunnableQuickPick[] = []
    if (prevRunnable) {
        items.push(prevRunnable)
    }
    for (let r of runnables) {
        if (prevRunnable && JSON.stringify(prevRunnable.runnable) == JSON.stringify(r)) {
            continue
        }
        items.push(new RunnableQuickPick(r))
    }
    let item = await vscode.window.showQuickPick(items)
    if (item) {
        item.detail = "rerun"
        prevRunnable = item
        let task = createTask(item.runnable)
        return await vscode.tasks.executeTask(task)
    }
}
