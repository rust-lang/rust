import * as vscode from 'vscode';
import * as toolchain from "./toolchain";
import { Config } from './config';
import { log } from './util';

// This ends up as the `type` key in tasks.json. RLS also uses `cargo` and
// our configuration should be compatible with it so use the same key.
export const TASK_TYPE = 'cargo';
export const TASK_SOURCE = 'rust';

export interface CargoTaskDefinition extends vscode.TaskDefinition {
    command?: string;
    args?: string[];
    cwd?: string;
    env?: { [key: string]: string };
    overrideCargo?: string;
}

class CargoTaskProvider implements vscode.TaskProvider {
    private readonly config: Config;

    constructor(config: Config) {
        this.config = config;
    }

    async provideTasks(): Promise<vscode.Task[]> {
        // Detect Rust tasks. Currently we do not do any actual detection
        // of tasks (e.g. aliases in .cargo/config) and just return a fixed
        // set of tasks that always exist. These tasks cannot be removed in
        // tasks.json - only tweaked.

        const defs = [
            { command: 'build', group: vscode.TaskGroup.Build },
            { command: 'check', group: vscode.TaskGroup.Build },
            { command: 'test', group: vscode.TaskGroup.Test },
            { command: 'clean', group: vscode.TaskGroup.Clean },
            { command: 'run', group: undefined },
        ];

        const tasks: vscode.Task[] = [];
        for (const workspaceTarget of vscode.workspace.workspaceFolders || []) {
            for (const def of defs) {
                const vscodeTask = await buildCargoTask(workspaceTarget, { type: TASK_TYPE, command: def.command }, `cargo ${def.command}`, [def.command], this.config.cargoRunner);
                vscodeTask.group = def.group;
                tasks.push(vscodeTask);
            }
        }

        return tasks;
    }

    async resolveTask(task: vscode.Task): Promise<vscode.Task | undefined> {
        // VSCode calls this for every cargo task in the user's tasks.json,
        // we need to inform VSCode how to execute that command by creating
        // a ShellExecution for it.

        const definition = task.definition as CargoTaskDefinition;

        if (definition.type === TASK_TYPE && definition.command) {
            const args = [definition.command].concat(definition.args ?? []);
            if (isWorkspaceFolder(task.scope)) {
                return await buildCargoTask(task.scope, definition, task.name, args, this.config.cargoRunner);
            }
        }

        return undefined;
    }
}

function isWorkspaceFolder(scope?: any): scope is vscode.WorkspaceFolder {
    return (scope as vscode.WorkspaceFolder).name !== undefined;
}

export async function buildCargoTask(
    target: vscode.WorkspaceFolder,
    definition: CargoTaskDefinition,
    name: string,
    args: string[],
    customRunner?: string,
    throwOnError: boolean = false
): Promise<vscode.Task> {

    let exec: vscode.ProcessExecution | vscode.ShellExecution | undefined = undefined;

    if (customRunner) {
        const runnerCommand = `${customRunner}.buildShellExecution`;
        try {
            const runnerArgs = { kind: TASK_TYPE, args, cwd: definition.cwd, env: definition.env };
            const customExec = await vscode.commands.executeCommand(runnerCommand, runnerArgs);
            if (customExec) {
                if (customExec instanceof vscode.ShellExecution) {
                    exec = customExec;
                } else {
                    log.debug("Invalid cargo ShellExecution", customExec);
                    throw "Invalid cargo ShellExecution.";
                }
            }
            // fallback to default processing

        } catch (e) {
            if (throwOnError) throw `Cargo runner '${customRunner}' failed! ${e}`;
            // fallback to default processing
        }
    }

    if (!exec) {
        // Check whether we must use a user-defined substitute for cargo.
        // Split on spaces to allow overrides like "wrapper cargo".
        const overrideCargo = definition.overrideCargo ?? definition.overrideCargo;
        const cargoCommand = overrideCargo?.split(" ") ?? [toolchain.cargoPath()];

        const fullCommand = [...cargoCommand, ...args];

        exec = new vscode.ProcessExecution(fullCommand[0], fullCommand.slice(1), definition);
    }

    return new vscode.Task(
        definition,
        target,
        name,
        TASK_SOURCE,
        exec,
        ['$rustc']
    );
}

export function activateTaskProvider(config: Config): vscode.Disposable {
    const provider = new CargoTaskProvider(config);
    return vscode.tasks.registerTaskProvider(TASK_TYPE, provider);
}
