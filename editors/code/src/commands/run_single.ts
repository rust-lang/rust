import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

interface Runnable {
    range: lc.Range;
    label: string;
    bin: string;
    args: string[];
    env: { [index: string]: string };
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
        label: 'cargo',
        command: spec.bin,
        args: spec.args,
        env: spec.env
    };

    const execOption: vscode.ShellExecutionOptions = {
        cwd: '.',
        env: definition.env
    };
    const exec = new vscode.ShellExecution(definition.command, definition.args, execOption);

    const f = vscode.workspace.workspaceFolders![0];
    const t = new vscode.Task(
        definition,
        f,
        definition.label,
        TASK_SOURCE,
        exec,
        ['$rustc']
    );
    t.presentationOptions.clear = true
    return t;
}

export async function handle(runnable: Runnable) {
    const editor = vscode.window.activeTextEditor;
    if (editor == null || editor.document.languageId !== 'rust') {
        return;
    }

    const task = createTask(runnable);
    task.group = vscode.TaskGroup.Build;
    task.presentationOptions = {
        reveal: vscode.TaskRevealKind.Always,
        panel: vscode.TaskPanelKind.Dedicated,
    };
    
    return vscode.tasks.executeTask(task);
}