import * as Is from "vscode-languageclient/lib/common/utils/is";
import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";
import type { Env } from "./client";
import { log } from "./util";
import { expectNotUndefined, unwrapUndefinable } from "./undefinable";

export type RunnableEnvCfg =
    | undefined
    | Record<string, string>
    | { mask?: string; env: Record<string, string> }[];

export class Config {
    readonly extensionId = "rust-lang.rust-analyzer";
    configureLang: vscode.Disposable | undefined;

    readonly rootSection = "rust-analyzer";
    private readonly requiresReloadOpts = [
        "cargo",
        "procMacro",
        "serverPath",
        "server",
        "files",
    ].map((opt) => `${this.rootSection}.${opt}`);

    readonly package: {
        version: string;
        releaseTag: string | null;
        enableProposedApi: boolean | undefined;
    } = vscode.extensions.getExtension(this.extensionId)!.packageJSON;

    readonly globalStorageUri: vscode.Uri;

    constructor(ctx: vscode.ExtensionContext) {
        this.globalStorageUri = ctx.globalStorageUri;
        this.discoveredWorkspaces = [];
        vscode.workspace.onDidChangeConfiguration(
            this.onDidChangeConfiguration,
            this,
            ctx.subscriptions,
        );
        this.refreshLogging();
        this.configureLanguage();
    }

    dispose() {
        this.configureLang?.dispose();
    }

    private refreshLogging() {
        log.setEnabled(this.traceExtension ?? false);
        log.info("Extension version:", this.package.version);

        const cfg = Object.entries(this.cfg).filter(([_, val]) => !(val instanceof Function));
        log.info("Using configuration", Object.fromEntries(cfg));
    }

    public discoveredWorkspaces: JsonProject[];

    private async onDidChangeConfiguration(event: vscode.ConfigurationChangeEvent) {
        this.refreshLogging();

        this.configureLanguage();

        const requiresReloadOpt = this.requiresReloadOpts.find((opt) =>
            event.affectsConfiguration(opt),
        );

        if (!requiresReloadOpt) return;

        if (this.restartServerOnConfigChange) {
            await vscode.commands.executeCommand("rust-analyzer.restartServer");
            return;
        }

        const message = `Changing "${requiresReloadOpt}" requires a server restart`;
        const userResponse = await vscode.window.showInformationMessage(message, "Restart now");

        if (userResponse) {
            const command = "rust-analyzer.restartServer";
            await vscode.commands.executeCommand(command);
        }
    }

    /**
     * Sets up additional language configuration that's impossible to do via a
     * separate language-configuration.json file. See [1] for more information.
     *
     * [1]: https://github.com/Microsoft/vscode/issues/11514#issuecomment-244707076
     */
    private configureLanguage() {
        // Only need to dispose of the config if there's a change
        if (this.configureLang) {
            this.configureLang.dispose();
            this.configureLang = undefined;
        }

        let onEnterRules: vscode.OnEnterRule[] = [
            {
                // Carry indentation from the previous line
                // if it's only whitespace
                beforeText: /^\s+$/,
                action: { indentAction: vscode.IndentAction.None },
            },
            {
                // After the end of a function/field chain,
                // with the semicolon on the same line
                beforeText: /^\s+\..*;/,
                action: { indentAction: vscode.IndentAction.Outdent },
            },
            {
                // After the end of a function/field chain,
                // with semicolon detached from the rest
                beforeText: /^\s+;/,
                previousLineText: /^\s+\..*/,
                action: { indentAction: vscode.IndentAction.Outdent },
            },
        ];

        if (this.typingContinueCommentsOnNewline) {
            const indentAction = vscode.IndentAction.None;

            onEnterRules = [
                ...onEnterRules,
                {
                    // Doc single-line comment
                    // e.g. ///|
                    beforeText: /^\s*\/{3}.*$/,
                    action: { indentAction, appendText: "/// " },
                },
                {
                    // Parent doc single-line comment
                    // e.g. //!|
                    beforeText: /^\s*\/{2}\!.*$/,
                    action: { indentAction, appendText: "//! " },
                },
                {
                    // Begins an auto-closed multi-line comment (standard or parent doc)
                    // e.g. /** | */ or /*! | */
                    beforeText: /^\s*\/\*(\*|\!)(?!\/)([^\*]|\*(?!\/))*$/,
                    afterText: /^\s*\*\/$/,
                    action: {
                        indentAction: vscode.IndentAction.IndentOutdent,
                        appendText: " * ",
                    },
                },
                {
                    // Begins a multi-line comment (standard or parent doc)
                    // e.g. /** ...| or /*! ...|
                    beforeText: /^\s*\/\*(\*|\!)(?!\/)([^\*]|\*(?!\/))*$/,
                    action: { indentAction, appendText: " * " },
                },
                {
                    // Continues a multi-line comment
                    // e.g.  * ...|
                    beforeText: /^(\ \ )*\ \*(\ ([^\*]|\*(?!\/))*)?$/,
                    action: { indentAction, appendText: "* " },
                },
                {
                    // Dedents after closing a multi-line comment
                    // e.g.  */|
                    beforeText: /^(\ \ )*\ \*\/\s*$/,
                    action: { indentAction, removeText: 1 },
                },
            ];
        }

        this.configureLang = vscode.languages.setLanguageConfiguration("rust", {
            onEnterRules,
        });
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
    private get<T>(path: string): T | undefined {
        return prepareVSCodeConfig(this.cfg.get<T>(path));
    }

    get serverPath() {
        return this.get<null | string>("server.path") ?? this.get<null | string>("serverPath");
    }

    get serverExtraEnv(): Env {
        const extraEnv =
            this.get<{ [key: string]: string | number } | null>("server.extraEnv") ?? {};
        return substituteVariablesInEnv(
            Object.fromEntries(
                Object.entries(extraEnv).map(([k, v]) => [
                    k,
                    typeof v !== "string" ? v.toString() : v,
                ]),
            ),
        );
    }
    get traceExtension() {
        return this.get<boolean>("trace.extension");
    }

    get discoverProjectCommand() {
        return this.get<string[] | undefined>("discoverProjectCommand");
    }

    get problemMatcher(): string[] {
        return this.get<string[]>("runnables.problemMatcher") || [];
    }

    get cargoRunner() {
        return this.get<string | undefined>("cargoRunner");
    }

    get runnablesExtraEnv() {
        const item = this.get<any>("runnables.extraEnv") ?? this.get<any>("runnableEnv");
        if (!item) return item;
        const fixRecord = (r: Record<string, any>) => {
            for (const key in r) {
                if (typeof r[key] !== "string") {
                    r[key] = String(r[key]);
                }
            }
        };
        if (item instanceof Array) {
            item.forEach((x) => fixRecord(x.env));
        } else {
            fixRecord(item);
        }
        return item;
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
                this.get<Record<string, string>>("debug.sourceFileMap") ?? {};
            sourceFileMap = trimmed;
        }

        return {
            engine: this.get<string>("debug.engine"),
            engineSettings: this.get<object>("debug.engineSettings") ?? {},
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
    get previewRustcOutput() {
        return this.get<boolean>("diagnostics.previewRustcOutput");
    }

    get useRustcErrorCode() {
        return this.get<boolean>("diagnostics.useRustcErrorCode");
    }

    get showDependenciesExplorer() {
        return this.get<boolean>("showDependenciesExplorer");
    }
}

// the optional `cb?` parameter is meant to be used to add additional
// key/value pairs to the VS Code configuration. This needed for, e.g.,
// including a `rust-project.json` into the `linkedProjects` key as part
// of the configuration/InitializationParams _without_ causing VS Code
// configuration to be written out to workspace-level settings. This is
// undesirable behavior because rust-project.json files can be tens of
// thousands of lines of JSON, most of which is not meant for humans
// to interact with.
export function prepareVSCodeConfig<T>(
    resp: T,
    cb?: (key: Extract<keyof T, string>, res: { [key: string]: any }) => void,
): T {
    if (Is.string(resp)) {
        return substituteVSCodeVariableInString(resp) as T;
    } else if (resp && Is.array<any>(resp)) {
        return resp.map((val) => {
            return prepareVSCodeConfig(val);
        }) as T;
    } else if (resp && typeof resp === "object") {
        const res: { [key: string]: any } = {};
        for (const key in resp) {
            const val = resp[key];
            res[key] = prepareVSCodeConfig(val);
            if (cb) {
                cb(key, res);
            }
        }
        return res as T;
    }
    return resp;
}

// FIXME: Merge this with `substituteVSCodeVariables` above
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
                const depName = unwrapUndefinable(match.groups?.["depName"]);
                deps.add(depName);
                // `depName` at this point can have a form of `expression` or
                // `prefix:expression`
                if (!definedEnvKeys.has(depName)) {
                    missingDeps.add(depName);
                }
            }
            return [`env:${key}`, { deps: [...deps], value }];
        }),
    );

    const resolved = new Set<string>();
    for (const dep of missingDeps) {
        const match = /(?<prefix>.*?):(?<body>.+)/.exec(dep);
        if (match) {
            const { prefix, body } = match.groups!;
            if (prefix === "env") {
                const envName = unwrapUndefinable(body);
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
                value: computeVscodeVar(dep) || "${" + dep + "}",
                deps: [],
            };
        }
    }
    const toResolve = new Set(Object.keys(envWithDeps));

    let leftToResolveSize;
    do {
        leftToResolveSize = toResolve.size;
        for (const key of toResolve) {
            const item = unwrapUndefinable(envWithDeps[key]);
            if (item.deps.every((dep) => resolved.has(dep))) {
                item.value = item.value.replace(/\${(?<depName>.+?)}/g, (_wholeMatch, depName) => {
                    const item = unwrapUndefinable(envWithDeps[depName]);
                    return item.value;
                });
                resolved.add(key);
                toResolve.delete(key);
            }
        }
    } while (toResolve.size > 0 && toResolve.size < leftToResolveSize);

    const resolvedEnv: Env = {};
    for (const key of Object.keys(env)) {
        const item = unwrapUndefinable(envWithDeps[`env:${key}`]);
        resolvedEnv[key] = item.value;
    }
    return resolvedEnv;
}

const VarRegex = new RegExp(/\$\{(.+?)\}/g);
function substituteVSCodeVariableInString(val: string): string {
    return val.replace(VarRegex, (substring: string, varName) => {
        if (Is.string(varName)) {
            return computeVscodeVar(varName) || substring;
        } else {
            return substring;
        }
    });
}

function computeVscodeVar(varName: string): string | null {
    const workspaceFolder = () => {
        const folders = vscode.workspace.workspaceFolders ?? [];
        const folder = folders[0];
        // TODO: support for remote workspaces?
        const fsPath: string =
            folder === undefined
                ? // no workspace opened
                  ""
                : // could use currently opened document to detect the correct
                  // workspace. However, that would be determined by the document
                  // user has opened on Editor startup. Could lead to
                  // unpredictable workspace selection in practice.
                  // It's better to pick the first one
                  folder.uri.fsPath;
        return fsPath;
    };
    // https://code.visualstudio.com/docs/editor/variables-reference
    const supportedVariables: { [k: string]: () => string } = {
        workspaceFolder,

        workspaceFolderBasename: () => {
            return path.basename(workspaceFolder());
        },

        cwd: () => process.cwd(),
        userHome: () => os.homedir(),

        // see
        // https://github.com/microsoft/vscode/blob/08ac1bb67ca2459496b272d8f4a908757f24f56f/src/vs/workbench/api/common/extHostVariableResolverService.ts#L81
        // or
        // https://github.com/microsoft/vscode/blob/29eb316bb9f154b7870eb5204ec7f2e7cf649bec/src/vs/server/node/remoteTerminalChannel.ts#L56
        execPath: () => process.env["VSCODE_EXEC_PATH"] ?? process.execPath,

        pathSeparator: () => path.sep,
    };

    if (varName in supportedVariables) {
        const fn = expectNotUndefined(
            supportedVariables[varName],
            `${varName} should not be undefined here`,
        );
        return fn();
    } else {
        // return "${" + varName + "}";
        return null;
    }
}
