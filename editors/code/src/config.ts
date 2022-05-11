import * as vscode from 'vscode';
import { Env } from './client';
import { log } from "./util";

export type UpdatesChannel = "stable" | "nightly";

const NIGHTLY_TAG = "nightly";

export type RunnableEnvCfg = undefined | Record<string, string> | { mask?: string; env: Record<string, string> }[];

export class Config {
    readonly extensionId = "matklad.rust-analyzer";

    readonly rootSection = "rust-analyzer";
    private readonly requiresReloadOpts = [
        "serverPath",
        "server",
        "cargo",
        "procMacro",
        "files",
        "lens", // works as lens.*
    ]
        .map(opt => `${this.rootSection}.${opt}`);

    readonly package: {
        version: string;
        releaseTag: string | null;
        enableProposedApi: boolean | undefined;
    } = vscode.extensions.getExtension(this.extensionId)!.packageJSON;

    readonly globalStorageUri: vscode.Uri;

    constructor(ctx: vscode.ExtensionContext) {
        this.globalStorageUri = ctx.globalStorageUri;
        vscode.workspace.onDidChangeConfiguration(this.onDidChangeConfiguration, this, ctx.subscriptions);
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

        const requiresReloadOpt = this.requiresReloadOpts.find(
            opt => event.affectsConfiguration(opt)
        );

        if (!requiresReloadOpt) return;

        const userResponse = await vscode.window.showInformationMessage(
            `Changing "${requiresReloadOpt}" requires a reload`,
            "Reload now"
        );

        if (userResponse === "Reload now") {
            await vscode.commands.executeCommand("workbench.action.reloadWindow");
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
    get serverExtraEnv() { return this.get<Env | null>("server.extraEnv") ?? {}; }
    get traceExtension() { return this.get<boolean>("trace.extension"); }

    get cargoRunner() {
        return this.get<string | undefined>("cargoRunner");
    }

    get runnableEnv() {
        return this.get<RunnableEnvCfg>("runnableEnv");
    }

    get debug() {
        let sourceFileMap = this.get<Record<string, string> | "auto">("debug.sourceFileMap");
        if (sourceFileMap !== "auto") {
            // "/rustc/<id>" used by suggestions only.
            const { ["/rustc/<id>"]: _, ...trimmed } = this.get<Record<string, string>>("debug.sourceFileMap");
            sourceFileMap = trimmed;
        }

        return {
            engine: this.get<string>("debug.engine"),
            engineSettings: this.get<object>("debug.engineSettings"),
            openDebugPane: this.get<boolean>("debug.openDebugPane"),
            sourceFileMap: sourceFileMap
        };
    }

    get hoverActions() {
        return {
            enable: this.get<boolean>("hoverActions.enable"),
            implementations: this.get<boolean>("hoverActions.implementations.enable"),
            references: this.get<boolean>("hoverActions.references.enable"),
            run: this.get<boolean>("hoverActions.run.enable"),
            debug: this.get<boolean>("hoverActions.debug.enable"),
            gotoTypeDef: this.get<boolean>("hoverActions.gotoTypeDef.enable"),
        };
    }

    get currentExtensionIsNightly() {
        return this.package.releaseTag === NIGHTLY_TAG;
    }
}

export async function updateConfig(config: vscode.WorkspaceConfiguration) {
    const renames = [
        ["assist.allowMergingIntoGlobImports", "imports.merge.glob",],
        ["assist.exprFillDefault", "assist.expressionFillDefault",],
        ["assist.importEnforceGranularity", "imports.granularity.enforce",],
        ["assist.importGranularity", "imports.granularity.group",],
        ["assist.importMergeBehavior", "imports.granularity.group",],
        ["assist.importMergeBehaviour", "imports.granularity.group",],
        ["assist.importGroup", "imports.group.enable",],
        ["assist.importPrefix", "imports.prefix",],
        ["cache.warmup", "primeCaches.enable",],
        ["cargo.loadOutDirsFromCheck", "cargo.buildScripts.enable",],
        ["cargo.runBuildScripts", "cargo.buildScripts.enable",],
        ["cargo.runBuildScriptsCommand", "cargo.buildScripts.overrideCommand",],
        ["cargo.useRustcWrapperForBuildScripts", "cargo.buildScripts.useRustcWrapper",],
        ["completion.snippets", "completion.snippets.custom",],
        ["diagnostics.enableExperimental", "diagnostics.experimental.enable",],
        ["experimental.procAttrMacros", "procMacro.attributes.enable",],
        ["highlighting.strings", "semanticHighlighting.strings.enable",],
        ["highlightRelated.breakPoints", "highlightRelated.breakPoints.enable",],
        ["highlightRelated.exitPoints", "highlightRelated.exitPoints.enable",],
        ["highlightRelated.yieldPoints", "highlightRelated.yieldPoints.enable",],
        ["highlightRelated.references", "highlightRelated.references.enable",],
        ["hover.documentation", "hover.documentation.enable",],
        ["hover.linksInHover", "hover.links.enable",],
        ["hoverActions.linksInHover", "hover.links.enable",],
        ["hoverActions.debug", "hoverActions.debug.enable",],
        ["hoverActions.enable", "hoverActions.enable.enable",],
        ["hoverActions.gotoTypeDef", "hoverActions.gotoTypeDef.enable",],
        ["hoverActions.implementations", "hoverActions.implementations.enable",],
        ["hoverActions.references", "hoverActions.references.enable",],
        ["hoverActions.run", "hoverActions.run.enable",],
        ["inlayHints.chainingHints", "inlayHints.chainingHints.enable",],
        ["inlayHints.closureReturnTypeHints", "inlayHints.closureReturnTypeHints.enable",],
        ["inlayHints.hideNamedConstructorHints", "inlayHints.typeHints.hideNamedConstructorHints",],
        ["inlayHints.parameterHints", "inlayHints.parameterHints.enable",],
        ["inlayHints.reborrowHints", "inlayHints.reborrowHints.enable",],
        ["inlayHints.typeHints", "inlayHints.typeHints.enable",],
        ["lruCapacity", "lru.capacity",],
        ["runnables.cargoExtraArgs", "runnables.extraArgs",],
        ["runnables.overrideCargo", "runnables.command",],
        ["rustcSource", "rustc.source",],
        ["rustfmt.enableRangeFormatting", "rustfmt.rangeFormatting.enable"]
    ];

    for (const [oldKey, newKey] of renames) {
        const inspect = config.inspect(oldKey);
        if (inspect !== undefined) {
            const valMatrix = [
                { val: inspect.globalValue, langVal: inspect.globalLanguageValue, target: vscode.ConfigurationTarget.Global },
                { val: inspect.workspaceFolderValue, langVal: inspect.workspaceFolderLanguageValue, target: vscode.ConfigurationTarget.WorkspaceFolder },
                { val: inspect.workspaceValue, langVal: inspect.workspaceLanguageValue, target: vscode.ConfigurationTarget.Workspace }
            ];
            for (const { val, langVal, target } of valMatrix) {
                const pred = (val: unknown) => {
                    // some of the updates we do only append "enable" or "custom"
                    // that means on the next run we would find these again, but as objects with
                    // these properties causing us to destroy the config
                    // so filter those already updated ones out
                    return val !== undefined && !(typeof val === "object" && val !== null && (val.hasOwnProperty("enable") || val.hasOwnProperty("custom")));
                };
                if (pred(val)) {
                    await config.update(newKey, val, target, false);
                    await config.update(oldKey, undefined, target, false);
                }
                if (pred(langVal)) {
                    await config.update(newKey, langVal, target, true);
                    await config.update(oldKey, undefined, target, true);
                }
            }
        }
    }
}
