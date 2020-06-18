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
}

class CargoTaskProvider implements vscode.TaskProvider {
    private readonly target: vscode.WorkspaceFolder;
    private readonly config: Config;

    constructor(target: vscode.WorkspaceFolder, config: Config) {
        this.target = target;
        this.config = config;
    }

    provideTasks(): vscode.Task[] {
        // Detect Rust tasks. Currently we do not do any actual detection
        // of tasks (e.g. aliases in .cargo/config) and just return a fixed
        // set of tasks that always exist. These tasks cannot be removed in
        // tasks.json - only tweaked.

        const cargoPath = toolchain.cargoPath();

        return [
            { command: 'build', group: vscode.TaskGroup.Build },
            { command: 'check', group: vscode.TaskGroup.Build },
            { command: 'test', group: vscode.TaskGroup.Test },
            { command: 'clean', group: vscode.TaskGroup.Clean },
            { command: 'run', group: undefined },
        ]
            .map(({ command, group }) => {
                const vscodeTask = new vscode.Task(
                    // The contents of this object end up in the tasks.json entries.
                    {
                        type: TASK_TYPE,
                        command,
                    },
                    // The scope of the task - workspace or specific folder (global
                    // is not supported).
                    this.target,
                    // The task name, and task source. These are shown in the UI as
                    // `${source}: ${name}`, e.g. `rust: cargo build`.
                    `cargo ${command}`,
                    'rust',
                    // What to do when this command is executed.
                    new vscode.ShellExecution(cargoPath, [command]),
                    // Problem matchers.
                    ['$rustc'],
                );
                vscodeTask.group = group;
                return vscodeTask;
            });
    }

    async resolveTask(task: vscode.Task): Promise<vscode.Task | undefined> {
        // VSCode calls this for every cargo task in the user's tasks.json,
        // we need to inform VSCode how to execute that command by creating
        // a ShellExecution for it.

        const definition = task.definition as CargoTaskDefinition;

        if (definition.type === TASK_TYPE && definition.command) {
            const args = [definition.command].concat(definition.args ?? []);

            return await buildCargoTask(definition, task.name, args, this.config.cargoRunner);
        }

        return undefined;
    }
}

export async function buildCargoTask(definition: CargoTaskDefinition, name: string, args: string[], customRunner?: string): Promise<vscode.Task> {
    if (customRunner) {
        const runnerCommand = `${customRunner}.createCargoTask`;
        try {
            const runnerArgs = { name, args, cwd: definition.cwd, env: definition.env, source: TASK_SOURCE };
            const task = await vscode.commands.executeCommand(runnerCommand, runnerArgs);

            if (task instanceof vscode.Task) {
                return task;
            } else if (task) {
                log.debug("Invalid cargo task", task);
                throw `Invalid task!`;
            }
            // fallback to default processing

        } catch (e) {
            throw `Cargo runner '${customRunner}' failed! ${e}`;
        }
    }

    return new vscode.Task(
        definition,
        name,
        TASK_SOURCE,
        new vscode.ShellExecution(toolchain.cargoPath(), args, definition),
    );
}

export function activateTaskProvider(target: vscode.WorkspaceFolder, config: Config): vscode.Disposable {
    const provider = new CargoTaskProvider(target, config);
    return vscode.tasks.registerTaskProvider(TASK_TYPE, provider);
}
