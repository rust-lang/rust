import type { Context } from ".";
import * as vscode from "vscode";
import * as assert from "assert";
import { targetToExecution } from "../../src/tasks";

export async function getTests(ctx: Context) {
    await ctx.suite("Tasks", (suite) => {
        suite.addTest("cargo targetToExecution", async () => {
            assert.deepStrictEqual(
                await targetToExecution({
                    type: "cargo",
                    command: "check",
                    args: ["foo"],
                }).then(executionToSimple),
                {
                    process: "cargo",
                    args: ["check", "foo"],
                },
            );
        });

        suite.addTest("shell targetToExecution", async () => {
            assert.deepStrictEqual(
                await targetToExecution({
                    type: "shell",
                    command: "thing",
                    args: ["foo"],
                }).then(executionToSimple),
                {
                    process: "thing",
                    args: ["foo"],
                },
            );
        });

        suite.addTest("base tasks", async () => {
            const tasks = await vscode.tasks.fetchTasks({ type: "cargo" });
            const expectedTasks = [
                {
                    definition: { type: "cargo", command: "build" },
                    name: "cargo build",
                    execution: {
                        process: "cargo",
                        args: ["build"],
                    },
                },
                {
                    definition: {
                        type: "cargo",
                        command: "check",
                    },
                    name: "cargo check",
                    execution: {
                        process: "cargo",
                        args: ["check"],
                    },
                },
                {
                    definition: { type: "cargo", command: "clippy" },
                    name: "cargo clippy",
                    execution: {
                        process: "cargo",
                        args: ["clippy"],
                    },
                },
                {
                    definition: { type: "cargo", command: "test" },
                    name: "cargo test",
                    execution: {
                        process: "cargo",
                        args: ["test"],
                    },
                },
                {
                    definition: {
                        type: "cargo",
                        command: "clean",
                    },
                    name: "cargo clean",
                    execution: {
                        process: "cargo",
                        args: ["clean"],
                    },
                },
                {
                    definition: { type: "cargo", command: "run" },
                    name: "cargo run",
                    execution: {
                        process: "cargo",
                        args: ["run"],
                    },
                },
            ];
            tasks.map(f).forEach((actual, i) => {
                const expected = expectedTasks[i];
                assert.deepStrictEqual(actual, expected);
            });
        });
    });
}

function f(task: vscode.Task): {
    definition: vscode.TaskDefinition;
    name: string;
    execution: {
        args: string[];
    } & ({ command: string } | { process: string });
} {
    const execution = executionToSimple(task.execution!);

    return {
        definition: task.definition,
        name: task.name,
        execution,
    };
}

function executionToSimple(
    taskExecution: vscode.ProcessExecution | vscode.ShellExecution | vscode.CustomExecution,
): {
    args: string[];
} & ({ command: string } | { process: string }) {
    const exec = taskExecution as vscode.ProcessExecution | vscode.ShellExecution;
    if (exec instanceof vscode.ShellExecution) {
        return {
            command: typeof exec.command === "string" ? exec.command : (exec.command?.value ?? ""),
            args: (exec.args ?? []).map((arg) => {
                if (typeof arg === "string") {
                    return arg;
                }
                return arg.value;
            }),
        };
    } else {
        return {
            process: exec.process,
            args: exec.args,
        };
    }
}
