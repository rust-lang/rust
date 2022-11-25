import path = require("path");
import * as vscode from "vscode";
import { Env } from "./client";
import { log } from "./util";

export type RunnableEnvCfg =
    | undefined
    | Record<string, string>
    | { mask?: string; env: Record<string, string> }[];

export class Config {
    readonly extensionId = "rust-lang.rust-analyzer";

    readonly rootSection = "rust-analyzer";
    private readonly requiresWorkspaceReloadOpts = [
        "serverPath",
        "server",
        // FIXME: This shouldn't be here, changing this setting should reload
        // `continueCommentsOnNewline` behavior without restart
        "typing",
    ].map((opt) => `${this.rootSection}.${opt}`);
    private readonly requiresReloadOpts = [
        "cargo",
        "procMacro",
        "files",
        "lens", // works as lens.*
    ]
        .map((opt) => `${this.rootSection}.${opt}`)
        .concat(this.requiresWorkspaceReloadOpts);

    readonly package: {
        version: string;
        releaseTag: string | null;
        enableProposedApi: boolean | undefined;
    } = vscode.extensions.getExtension(this.extensionId)!.packageJSON;

    readonly globalStorageUri: vscode.Uri;

    constructor(ctx: vscode.ExtensionContext) {
        this.globalStorageUri = ctx.globalStorageUri;
        vscode.workspace.onDidChangeConfiguration(
            this.onDidChangeConfiguration,
            this,
            ctx.subscriptions
        );
        this.refreshLogging();
    }

    private refreshLogging() {
        log.setEnabled(this.traceExtension);
        log.info("Extension version:", this.package.version);

        const cfg = Object.entries(this.cfg).filter(([_, val]) => !(val instanceof Function));
        log.info("Using configuration", Object.fromEntries(cfg));
    }

    private async onDidChangeConfiguration(event: vscode.ConfigurationChangeEvent) {
        this.refreshLogging();

        const requiresReloadOpt = this.requiresReloadOpts.find((opt) =>
            event.affectsConfiguration(opt)
        );

        if (!requiresReloadOpt) return;

        const requiresWorkspaceReloadOpt = this.requiresWorkspaceReloadOpts.find((opt) =>
            event.affectsConfiguration(opt)
        );

        if (!requiresWorkspaceReloadOpt && this.restartServerOnConfigChange) {
            await vscode.commands.executeCommand("rust-analyzer.reload");
            return;
        }

        const message = requiresWorkspaceReloadOpt
            ? `Changing "${requiresWorkspaceReloadOpt}" requires a window reload`
            : `Changing "${requiresReloadOpt}" requires a reload`;
        const userResponse = await vscode.window.showInformationMessage(message, "Reload now");

        if (userResponse === "Reload now") {
            const command = requiresWorkspaceReloadOpt
                ? "workbench.action.reloadWindow"
                : "rust-analyzer.reload";
            if (userResponse === "Reload now") {
                await vscode.commands.executeCommand(command);
            }
        }
    }

    // We don't do runtime config validation here for simplicity. More on stackoverflow:
    // https://stackoverflow.com/questions/60135780/what-is-the-best-way-to-type-check-the-configuration-for-vscode-extension

    private get cfg(): vscode.WorkspaceConfiguration {
        return vscode.workspace.getConfiguration(this.rootSection);
    }

    /**
     * Beware that postfix `!` operator erases both `null` and `undefined`.
     * This is why the following doesn't work as expected:
     *
     * ```ts
     * const nullableNum = vscode
     *  .workspace
     *  .getConfiguration
     *  .getConfiguration("rust-analyzer")
     *  .get<number | null>(path)!;
     *
     * // What happens is that type of `nullableNum` is `number` but not `null | number`:
     * const fullFledgedNum: number = nullableNum;
     * ```
     * So this getter handles this quirk by not requiring the caller to use postfix `!`
     */
    private get<T>(path: string): T {
        return this.cfg.get<T>(path)!;
    }

    get serverPath() {
        return this.get<null | string>("server.path") ?? this.get<null | string>("serverPath");
    }
    get serverExtraEnv(): Env {
        const extraEnv =
            this.get<{ [key: string]: string | number } | null>("server.extraEnv") ?? {};
        return Object.fromEntries(
            Object.entries(extraEnv).map(([k, v]) => [k, typeof v !== "string" ? v.toString() : v])
        );
    }
    get traceExtension() {
        return this.get<boolean>("trace.extension");
    }

    get cargoRunner() {
        return this.get<string | undefined>("cargoRunner");
    }

    get runnableEnv() {
        return this.get<RunnableEnvCfg>("runnableEnv");
    }

    get restartServerOnConfigChange() {
        return this.get<boolean>("restartServerOnConfigChange");
    }

    get typingContinueCommentsOnNewline() {
        return this.get<boolean>("typing.continueCommentsOnNewline");
    }

    get debug() {
        let sourceFileMap = this.get<Record<string, string> | "auto">("debug.sourceFileMap");
        if (sourceFileMap !== "auto") {
            // "/rustc/<id>" used by suggestions only.
            const { ["/rustc/<id>"]: _, ...trimmed } =
                this.get<Record<string, string>>("debug.sourceFileMap");
            sourceFileMap = trimmed;
        }

        return {
            engine: this.get<string>("debug.engine"),
            engineSettings: this.get<object>("debug.engineSettings"),
            openDebugPane: this.get<boolean>("debug.openDebugPane"),
            sourceFileMap: sourceFileMap,
        };
    }

    get hoverActions() {
        return {
            enable: this.get<boolean>("hover.actions.enable"),
            implementations: this.get<boolean>("hover.actions.implementations.enable"),
            references: this.get<boolean>("hover.actions.references.enable"),
            run: this.get<boolean>("hover.actions.run.enable"),
            debug: this.get<boolean>("hover.actions.debug.enable"),
            gotoTypeDef: this.get<boolean>("hover.actions.gotoTypeDef.enable"),
        };
    }
}

export function substituteVariablesInEnv(env: Env): Env {
    const missingDeps = new Set<string>();
    // vscode uses `env:ENV_NAME` for env vars resolution, and it's easier
    // to follow the same convention for our dependency tracking
    const definedEnvKeys = new Set(Object.keys(env).map((key) => `env:${key}`));
    const envWithDeps = Object.fromEntries(
        Object.entries(env).map(([key, value]) => {
            const deps = new Set<string>();
            const depRe = new RegExp(/\${(?<depName>.+?)}/g);
            let match = undefined;
            while ((match = depRe.exec(value))) {
                const depName = match.groups!.depName;
                deps.add(depName);
                // `depName` at this point can have a form of `expression` or
                // `prefix:expression`
                if (!definedEnvKeys.has(depName)) {
                    missingDeps.add(depName);
                }
            }
            return [`env:${key}`, { deps: [...deps], value }];
        })
    );

    const resolved = new Set<string>();
    for (const dep of missingDeps) {
        const match = /(?<prefix>.*?):(?<body>.+)/.exec(dep);
        if (match) {
            const { prefix, body } = match.groups!;
            if (prefix === "env") {
                const envName = body;
                envWithDeps[dep] = {
                    value: process.env[envName] ?? "",
                    deps: [],
                };
                resolved.add(dep);
            } else {
                // we can't handle other prefixes at the moment
                // leave values as is, but still mark them as resolved
                envWithDeps[dep] = {
                    value: "${" + dep + "}",
                    deps: [],
                };
                resolved.add(dep);
            }
        } else {
            envWithDeps[dep] = {
                value: computeVscodeVar(dep),
                deps: [],
            };
        }
    }
    const toResolve = new Set(Object.keys(envWithDeps));

    let leftToResolveSize;
    do {
        leftToResolveSize = toResolve.size;
        for (const key of toResolve) {
            if (envWithDeps[key].deps.every((dep) => resolved.has(dep))) {
                envWithDeps[key].value = envWithDeps[key].value.replace(
                    /\${(?<depName>.+?)}/g,
                    (_wholeMatch, depName) => {
                        return envWithDeps[depName].value;
                    }
                );
                resolved.add(key);
                toResolve.delete(key);
            }
        }
    } while (toResolve.size > 0 && toResolve.size < leftToResolveSize);

    const resolvedEnv: Env = {};
    for (const key of Object.keys(env)) {
        resolvedEnv[key] = envWithDeps[`env:${key}`].value;
    }
    return resolvedEnv;
}

function computeVscodeVar(varName: string): string {
    // https://code.visualstudio.com/docs/editor/variables-reference
    const supportedVariables: { [k: string]: () => string } = {
        workspaceFolder: () => {
            const folders = vscode.workspace.workspaceFolders ?? [];
            if (folders.length === 1) {
                // TODO: support for remote workspaces?
                return folders[0].uri.fsPath;
            } else if (folders.length > 1) {
                // could use currently opened document to detect the correct
                // workspace. However, that would be determined by the document
                // user has opened on Editor startup. Could lead to
                // unpredictable workspace selection in practice.
                // It's better to pick the first one
                return folders[0].uri.fsPath;
            } else {
                // no workspace opened
                return "";
            }
        },

        workspaceFolderBasename: () => {
            const workspaceFolder = computeVscodeVar("workspaceFolder");
            if (workspaceFolder) {
                return path.basename(workspaceFolder);
            } else {
                return "";
            }
        },

        cwd: () => process.cwd(),

        // see
        // https://github.com/microsoft/vscode/blob/08ac1bb67ca2459496b272d8f4a908757f24f56f/src/vs/workbench/api/common/extHostVariableResolverService.ts#L81
        // or
        // https://github.com/microsoft/vscode/blob/29eb316bb9f154b7870eb5204ec7f2e7cf649bec/src/vs/server/node/remoteTerminalChannel.ts#L56
        execPath: () => process.env.VSCODE_EXEC_PATH ?? process.execPath,

        pathSeparator: () => path.sep,
    };

    if (varName in supportedVariables) {
        return supportedVariables[varName]();
    } else {
        // can't resolve, keep the expression as is
        return "${" + varName + "}";
    }
}
