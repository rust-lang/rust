import * as os from "os";
import * as vscode from "vscode";
import * as path from "path";
import type * as ra from "./lsp_ext";

import { Cargo } from "./toolchain";
import type { Ctx } from "./ctx";
import { createTaskFromRunnable, prepareEnv } from "./run";
import { execute, isCargoRunnableArgs, unwrapUndefinable } from "./util";
import type { Config } from "./config";

const debugOutput = vscode.window.createOutputChannel("Debug");

// Here we want to keep track on everything that's currently running
const activeDebugSessionIds: string[] = [];

export async function makeDebugConfig(ctx: Ctx, runnable: ra.Runnable): Promise<void> {
    const scope = ctx.activeRustEditor?.document.uri;
    if (!scope) return;

    const debugConfig = await getDebugConfiguration(ctx.config, runnable, false);
    if (!debugConfig) return;

    const wsLaunchSection = vscode.workspace.getConfiguration("launch", scope);
    const configurations = wsLaunchSection.get<any[]>("configurations") || [];

    const index = configurations.findIndex((c) => c.name === debugConfig.name);
    if (index !== -1) {
        const answer = await vscode.window.showErrorMessage(
            `Launch configuration '${debugConfig.name}' already exists!`,
            "Cancel",
            "Update",
        );
        if (answer === "Cancel") return;

        configurations[index] = debugConfig;
    } else {
        configurations.push(debugConfig);
    }

    await wsLaunchSection.update("configurations", configurations);
}

export async function startDebugSession(ctx: Ctx, runnable: ra.Runnable): Promise<boolean> {
    let debugConfig: vscode.DebugConfiguration | undefined = undefined;
    let message = "";

    const wsLaunchSection = vscode.workspace.getConfiguration("launch");
    const configurations = wsLaunchSection.get<any[]>("configurations") || [];

    // The runnable label is the name of the test with the "test prefix"
    // e.g. test test_feature_x
    const index = configurations.findIndex((c) => c.name === runnable.label);
    if (-1 !== index) {
        debugConfig = configurations[index];
        message = " (from launch.json)";
        debugOutput.clear();
    } else {
        debugConfig = await getDebugConfiguration(ctx.config, runnable);
    }

    if (!debugConfig) return false;

    debugOutput.appendLine(`Launching debug configuration${message}:`);
    debugOutput.appendLine(JSON.stringify(debugConfig, null, 2));
    return vscode.debug.startDebugging(undefined, debugConfig);
}

function createCommandLink(extensionId: string): string {
    // do not remove the second quotes inside
    // encodeURIComponent or it won't work
    return `extension.open?${encodeURIComponent(`"${extensionId}"`)}`;
}

async function getDebugConfiguration(
    config: Config,
    runnable: ra.Runnable,
    inheritEnv: boolean = true,
): Promise<vscode.DebugConfiguration | undefined> {
    if (!isCargoRunnableArgs(runnable.args)) {
        return;
    }
    const runnableArgs: ra.CargoRunnableArgs = runnable.args;

    const debugOptions = config.debug;

    let provider: null | KnownEnginesType = null;

    if (debugOptions.engine === "auto") {
        for (const engineId in knownEngines) {
            const debugEngine = vscode.extensions.getExtension(engineId);
            if (debugEngine) {
                provider = knownEngines[engineId as keyof typeof knownEngines];
                break;
            }
        }
    } else if (debugOptions.engine) {
        const debugEngine = vscode.extensions.getExtension(debugOptions.engine);
        if (debugEngine && Object.keys(knownEngines).includes(debugOptions.engine)) {
            provider = knownEngines[debugOptions.engine as keyof typeof knownEngines];
        }
    }

    if (!provider) {
        const commandCCpp: string = createCommandLink("ms-vscode.cpptools");
        const commandCodeLLDB: string = createCommandLink("vadimcn.vscode-lldb");
        const commandNativeDebug: string = createCommandLink("webfreak.debug");

        await vscode.window.showErrorMessage(
            `Install [CodeLLDB](command:${commandCodeLLDB} "Open CodeLLDB")` +
                `, [C/C++](command:${commandCCpp} "Open C/C++") ` +
                `or [Native Debug](command:${commandNativeDebug} "Open Native Debug") for debugging.`,
        );
        return;
    }

    debugOutput.clear();
    if (config.debug.openDebugPane) {
        debugOutput.show(true);
    }
    // folder exists or RA is not active.
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-type-assertion
    const workspaceFolders = vscode.workspace.workspaceFolders!;
    const isMultiFolderWorkspace = workspaceFolders.length > 1;
    const firstWorkspace = workspaceFolders[0];
    const maybeWorkspace =
        !isMultiFolderWorkspace || !runnableArgs.workspaceRoot
            ? firstWorkspace
            : workspaceFolders.find((w) => runnableArgs.workspaceRoot?.includes(w.uri.fsPath)) ||
              firstWorkspace;

    const workspace = unwrapUndefinable(maybeWorkspace);
    let wsFolder = path.normalize(workspace.uri.fsPath);
    if (os.platform() === "win32") {
        // in windows, the drive letter can vary in casing for VSCode, so we gotta normalize that first
        wsFolder = wsFolder.replace(/^[a-z]:\\/, (c) => c.toUpperCase());
    }

    const workspaceQualifier = isMultiFolderWorkspace ? `:${workspace.name}` : "";
    function simplifyPath(p: string): string {
        // in windows, the drive letter can vary in casing for VSCode, so we gotta normalize that first
        if (os.platform() === "win32") {
            p = p.replace(/^[a-z]:\\/, (c) => c.toUpperCase());
        }
        // see https://github.com/rust-lang/rust-analyzer/pull/5513#issuecomment-663458818 for why this is needed
        return path.normalize(p).replace(wsFolder, `\${workspaceFolder${workspaceQualifier}}`);
    }

    const env = prepareEnv(inheritEnv, runnable.label, runnableArgs, config.runnablesExtraEnv);
    const executable = await getDebugExecutable(runnableArgs, env);
    let sourceFileMap = debugOptions.sourceFileMap;
    if (sourceFileMap === "auto") {
        sourceFileMap = {};
        await discoverSourceFileMap(sourceFileMap, env, wsFolder);
    }

    const debugConfig = getDebugConfig(
        provider,
        simplifyPath,
        runnable,
        runnableArgs,
        executable,
        env,
        sourceFileMap,
    );
    if (debugConfig.type in debugOptions.engineSettings) {
        const settingsMap = (debugOptions.engineSettings as any)[debugConfig.type];
        for (var key in settingsMap) {
            debugConfig[key] = settingsMap[key];
        }
    }

    if (debugConfig.name === "run binary") {
        // The LSP side: crates\rust-analyzer\src\main_loop\handlers.rs,
        // fn to_lsp_runnable(...) with RunnableKind::Bin
        // FIXME: Neither crates\rust-analyzer\src\main_loop\handlers.rs
        // nor to_lsp_runnable exist anymore
        debugConfig.name = `run ${path.basename(executable)}`;
    }

    const cwd = debugConfig["cwd"];
    if (cwd) {
        debugConfig["cwd"] = simplifyPath(cwd);
    }

    return debugConfig;
}

async function discoverSourceFileMap(
    sourceFileMap: Record<string, string>,
    env: Record<string, string>,
    cwd: string,
) {
    const sysroot = env["RUSTC_TOOLCHAIN"];
    if (sysroot) {
        // let's try to use the default toolchain
        const data = await execute(`rustc -V -v`, { cwd, env });
        const rx = /commit-hash:\s(.*)$/m;

        const commitHash = rx.exec(data)?.[1];
        if (commitHash) {
            const rustlib = path.normalize(sysroot + "/lib/rustlib/src/rust");
            sourceFileMap[`/rustc/${commitHash}/`] = rustlib;
        }
    }
}

type PropertyFetcher<Config, Input, Key extends keyof Config> = (
    input: Input,
) => [Key, Config[Key]];

type DebugConfigProvider<Type extends string, DebugConfig extends BaseDebugConfig<Type>> = {
    executableProperty: keyof DebugConfig;
    environmentProperty: PropertyFetcher<DebugConfig, Record<string, string>, keyof DebugConfig>;
    runnableArgsProperty: PropertyFetcher<DebugConfig, ra.CargoRunnableArgs, keyof DebugConfig>;
    sourceFileMapProperty?: keyof DebugConfig;
    type: Type;
    additional?: Record<string, unknown>;
};

type KnownEnginesType = (typeof knownEngines)[keyof typeof knownEngines];
const knownEngines: {
    "vadimcn.vscode-lldb": DebugConfigProvider<"lldb", CodeLldbDebugConfig>;
    "ms-vscode.cpptools": DebugConfigProvider<"cppvsdbg" | "cppdbg", CCppDebugConfig>;
    "webfreak.debug": DebugConfigProvider<"gdb", NativeDebugConfig>;
} = {
    "vadimcn.vscode-lldb": {
        type: "lldb",
        executableProperty: "program",
        environmentProperty: (env) => ["env", env],
        runnableArgsProperty: (runnableArgs: ra.CargoRunnableArgs) => [
            "args",
            runnableArgs.executableArgs,
        ],
        sourceFileMapProperty: "sourceMap",
        additional: {
            sourceLanguages: ["rust"],
        },
    },
    "ms-vscode.cpptools": {
        type: os.platform() === "win32" ? "cppvsdbg" : "cppdbg",
        executableProperty: "program",
        environmentProperty: (env) => [
            "environment",
            Object.entries(env).map((entry) => ({
                name: entry[0],
                value: entry[1],
            })),
        ],
        runnableArgsProperty: (runnableArgs: ra.CargoRunnableArgs) => [
            "args",
            runnableArgs.executableArgs,
        ],
        sourceFileMapProperty: "sourceFileMap",
        additional: {
            osx: {
                MIMode: "lldb",
            },
        },
    },
    "webfreak.debug": {
        type: "gdb",
        executableProperty: "target",
        runnableArgsProperty: (runnableArgs: ra.CargoRunnableArgs) => [
            "arguments",
            quote(runnableArgs.executableArgs),
        ],
        environmentProperty: (env) => ["env", env],
        additional: {
            valuesFormatting: "prettyPrinters",
        },
    },
};

async function getDebugExecutable(
    runnableArgs: ra.CargoRunnableArgs,
    env: Record<string, string>,
): Promise<string> {
    const cargo = new Cargo(runnableArgs.workspaceRoot || ".", debugOutput, env);
    const executable = await cargo.executableFromArgs(runnableArgs);

    // if we are here, there were no compilation errors.
    return executable;
}

type BaseDebugConfig<type extends string> = {
    type: type;
    request: "launch";
    name: string;
    cwd: string;
};

function getDebugConfig(
    provider: KnownEnginesType,
    simplifyPath: (p: string) => string,
    runnable: ra.Runnable,
    runnableArgs: ra.CargoRunnableArgs,
    executable: string,
    env: Record<string, string>,
    sourceFileMap?: Record<string, string>,
): vscode.DebugConfiguration {
    const {
        environmentProperty,
        executableProperty,
        runnableArgsProperty,
        type,
        additional,
        sourceFileMapProperty,
    } = provider;
    const [envProperty, envValue] = environmentProperty(env);
    const [argsProperty, argsValue] = runnableArgsProperty(runnableArgs);
    return {
        type,
        request: "launch",
        name: runnable.label,
        cwd: simplifyPath(runnable.args.cwd || runnableArgs.workspaceRoot || "."),
        [executableProperty]: simplifyPath(executable),
        [envProperty]: envValue,
        [argsProperty]: argsValue,
        ...(sourceFileMapProperty ? { [sourceFileMapProperty]: sourceFileMap } : {}),
        ...additional,
    };
}

type CCppDebugConfig = {
    program: string;
    args: string[];
    sourceFileMap: Record<string, string> | undefined;
    environment: {
        name: string;
        value: string;
    }[];
    // See https://github.com/rust-lang/rust-analyzer/issues/16901#issuecomment-2024486941
    osx: {
        MIMode: "lldb";
    };
} & BaseDebugConfig<"cppvsdbg" | "cppdbg">;

type CodeLldbDebugConfig = {
    program: string;
    args: string[];
    sourceMap: Record<string, string> | undefined;
    sourceLanguages: ["rust"];
    env: Record<string, string>;
} & BaseDebugConfig<"lldb">;

type NativeDebugConfig = {
    target: string;
    // See https://github.com/WebFreak001/code-debug/issues/359
    arguments: string;
    env: Record<string, string>;
    valuesFormatting: "prettyPrinters";
} & BaseDebugConfig<"gdb">;

// Based on https://github.com/ljharb/shell-quote/blob/main/quote.js
function quote(xs: string[]) {
    return xs
        .map(function (s) {
            if (/["\s]/.test(s) && !/'/.test(s)) {
                return "'" + s.replace(/(['\\])/g, "\\$1") + "'";
            }
            if (/["'\s]/.test(s)) {
                return '"' + s.replace(/(["\\$`!])/g, "\\$1") + '"';
            }
            return s.replace(/([A-Za-z]:)?([#!"$&'()*,:;<=>?@[\\\]^`{|}])/g, "$1\\$2");
        })
        .join(" ");
}

async function recompileTestFromDebuggingSession(session: vscode.DebugSession, ctx: Ctx) {
    const { cwd, args: sessionArgs }: vscode.DebugConfiguration = session.configuration;

    const args: ra.CargoRunnableArgs = {
        cwd: cwd,
        cargoArgs: ["test", "--no-run", "--test", "lib"],

        // The first element of the debug configuration args is the test path e.g. "test_bar::foo::test_a::test_b"
        executableArgs: sessionArgs,
    };
    const runnable: ra.Runnable = {
        kind: "cargo",
        label: "compile-test",
        args,
    };
    const task: vscode.Task = await createTaskFromRunnable(runnable, ctx.config);

    // It is not needed to call the language server, since the test path is already resolved in the
    // configuration option. We can simply call a debug configuration with the --no-run option to compile
    await vscode.tasks.executeTask(task);
}

export function initializeDebugSessionTrackingAndRebuild(ctx: Ctx) {
    vscode.debug.onDidStartDebugSession((session: vscode.DebugSession) => {
        if (!activeDebugSessionIds.includes(session.id)) {
            activeDebugSessionIds.push(session.id);
        }
    });

    vscode.debug.onDidTerminateDebugSession(async (session: vscode.DebugSession) => {
        // The id of the session will be the same when pressing restart the restart button
        if (activeDebugSessionIds.find((s) => s === session.id)) {
            await recompileTestFromDebuggingSession(session, ctx);
        }
        removeActiveSession(session);
    });
}

function removeActiveSession(session: vscode.DebugSession) {
    const activeSessionId = activeDebugSessionIds.findIndex((id) => id === session.id);

    if (activeSessionId !== -1) {
        activeDebugSessionIds.splice(activeSessionId, 1);
    }
}
