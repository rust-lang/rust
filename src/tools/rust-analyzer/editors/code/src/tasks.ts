import * as vscode from "vscode";
import type { Config } from "./config";
import { log, unwrapUndefinable } from "./util";
import * as toolchain from "./toolchain";

// This ends up as the `type` key in tasks.json. RLS also uses `cargo` and
// our configuration should be compatible with it so use the same key.
export const CARGO_TASK_TYPE = "cargo";
export const SHELL_TASK_TYPE = "shell";

export const RUST_TASK_SOURCE = "rust";

export type RustTargetDefinition = {
    readonly type: typeof CARGO_TASK_TYPE | typeof SHELL_TASK_TYPE;
} & vscode.TaskDefinition &
    RustTarget;
export type RustTarget = {
    // The command to run, usually `cargo`.
    command: string;
    // Additional arguments passed to the command.
    args?: string[];
    // The working directory to run the command in.
    cwd?: string;
    // The shell environment.
    env?: { [key: string]: string };
};

class RustTaskProvider implements vscode.TaskProvider {
    private readonly config: Config;

    constructor(config: Config) {
        this.config = config;
    }

    async provideTasks(): Promise<vscode.Task[]> {
        if (!vscode.workspace.workspaceFolders) {
            return [];
        }

        // Detect Rust tasks. Currently we do not do any actual detection
        // of tasks (e.g. aliases in .cargo/config) and just return a fixed
        // set of tasks that always exist. These tasks cannot be removed in
        // tasks.json - only tweaked.

        const defs = [
            { command: "build", group: vscode.TaskGroup.Build },
            { command: "check", group: vscode.TaskGroup.Build },
            { command: "clippy", group: vscode.TaskGroup.Build },
            { command: "test", group: vscode.TaskGroup.Test },
            { command: "clean", group: vscode.TaskGroup.Clean },
            { command: "run", group: undefined },
        ];

        // FIXME: The server should provide this
        const cargo = await toolchain.cargoPath();

        const tasks: vscode.Task[] = [];
        for (const workspaceTarget of vscode.workspace.workspaceFolders) {
            for (const def of defs) {
                const definition = {
                    command: cargo,
                    args: [def.command],
                };
                const exec = await targetToExecution(definition, this.config.cargoRunner);
                const vscodeTask = await buildRustTask(
                    workspaceTarget,
                    { ...definition, type: CARGO_TASK_TYPE },
                    `cargo ${def.command}`,
                    this.config.problemMatcher,
                    exec,
                );
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
        if (task.definition.type === CARGO_TASK_TYPE) {
            const taskDefinition = task.definition as RustTargetDefinition;
            const cargo = await toolchain.cargoPath();
            const exec = await targetToExecution(
                {
                    command: cargo,
                    args: [taskDefinition.command].concat(taskDefinition.args || []),
                    cwd: taskDefinition.cwd,
                    env: taskDefinition.env,
                },
                this.config.cargoRunner,
            );
            return await buildRustTask(
                task.scope,
                taskDefinition,
                task.name,
                this.config.problemMatcher,
                exec,
            );
        }

        return undefined;
    }
}

export async function buildRustTask(
    scope: vscode.WorkspaceFolder | vscode.TaskScope | undefined,
    definition: RustTargetDefinition,
    name: string,
    problemMatcher: string[],
    exec: vscode.ProcessExecution | vscode.ShellExecution,
): Promise<vscode.Task> {
    return new vscode.Task(
        definition,
        // scope can sometimes be undefined. in these situations we default to the workspace taskscope as
        // recommended by the official docs: https://code.visualstudio.com/api/extension-guides/task-provider#task-provider)
        scope ?? vscode.TaskScope.Workspace,
        name,
        RUST_TASK_SOURCE,
        exec,
        problemMatcher,
    );
}

export async function targetToExecution(
    definition: RustTarget,
    customRunner?: string,
    throwOnError: boolean = false,
): Promise<vscode.ProcessExecution | vscode.ShellExecution> {
    if (customRunner) {
        const runnerCommand = `${customRunner}.buildShellExecution`;

        try {
            const runnerArgs = {
                kind: CARGO_TASK_TYPE,
                args: definition.args,
                cwd: definition.cwd,
                env: definition.env,
            };
            const customExec = await vscode.commands.executeCommand(runnerCommand, runnerArgs);
            if (customExec) {
                if (customExec instanceof vscode.ShellExecution) {
                    return customExec;
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
    const args = unwrapUndefinable(definition.args);
    return new vscode.ProcessExecution(definition.command, args, {
        cwd: definition.cwd,
        env: definition.env,
    });
}

export function activateTaskProvider(config: Config): vscode.Disposable {
    const provider = new RustTaskProvider(config);
    return vscode.tasks.registerTaskProvider(CARGO_TASK_TYPE, provider);
}
