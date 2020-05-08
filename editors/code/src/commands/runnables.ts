import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';
import * as ra from '../rust-analyzer-api';
import * as os from "os";

import { Ctx, Cmd } from '../ctx';
import { Cargo } from '../cargo';

export function run(ctx: Ctx): Cmd {
    let prevRunnable: RunnableQuickPick | undefined;

    return async () => {
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
        const item = await vscode.window.showQuickPick(items);
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

function getLldbDebugConfig(config: ra.Runnable, executable: string, sourceFileMap?: Record<string, string>): vscode.DebugConfiguration {
    return {
        type: "lldb",
        request: "launch",
        name: config.label,
        program: executable,
        args: config.extraArgs,
        cwd: config.cwd,
        sourceMap: sourceFileMap,
        sourceLanguages: ["rust"]
    };
}

function getCppvsDebugConfig(config: ra.Runnable, executable: string, sourceFileMap?: Record<string, string>): vscode.DebugConfiguration {
    return {
        type: (os.platform() === "win32") ? "cppvsdbg" : 'cppdbg',
        request: "launch",
        name: config.label,
        program: executable,
        args: config.extraArgs,
        cwd: config.cwd,
        sourceFileMap: sourceFileMap,
    };
}

const debugOutput = vscode.window.createOutputChannel("Debug");

async function getDebugExecutable(config: ra.Runnable): Promise<string> {
    const cargo = new Cargo(config.cwd || '.', debugOutput);
    const executable = await cargo.executableFromArgs(config.args);

    // if we are here, there were no compilation errors.
    return executable;
}

type DebugConfigProvider = (config: ra.Runnable, executable: string, sourceFileMap?: Record<string, string>) => vscode.DebugConfiguration;

export function debugSingle(ctx: Ctx): Cmd {
    return async (config: ra.Runnable) => {
        const editor = ctx.activeRustEditor;
        if (!editor) return;

        const knownEngines: Record<string, DebugConfigProvider> = {
            "vadimcn.vscode-lldb": getLldbDebugConfig,
            "ms-vscode.cpptools": getCppvsDebugConfig
        };
        const debugOptions = ctx.config.debug;

        let debugEngine = null;
        if (debugOptions.engine === "auto") {
            for (var engineId in knownEngines) {
                debugEngine = vscode.extensions.getExtension(engineId);
                if (debugEngine) break;
            }
        }
        else {
            debugEngine = vscode.extensions.getExtension(debugOptions.engine);
        }

        if (!debugEngine) {
            vscode.window.showErrorMessage(`Install [CodeLLDB](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb)`
                + ` or [MS C++ tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) extension for debugging.`);
            return;
        }

        debugOutput.clear();
        if (ctx.config.debug.openUpDebugPane) {
            debugOutput.show(true);
        }

        const executable = await getDebugExecutable(config);
        const debugConfig = knownEngines[debugEngine.id](config, executable, debugOptions.sourceFileMap);
        if (debugConfig.type in debugOptions.engineSettings) {
            const settingsMap = (debugOptions.engineSettings as any)[debugConfig.type];
            for (var key in settingsMap) {
                debugConfig[key] = settingsMap[key];
            }
        }

        debugOutput.appendLine("Launching debug configuration:");
        debugOutput.appendLine(JSON.stringify(debugConfig, null, 2));
        return vscode.debug.startDebugging(undefined, debugConfig);
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
