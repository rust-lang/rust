import * as vscode from 'vscode';
import { log } from "./util";

export type UpdatesChannel = "stable" | "nightly";

export const NIGHTLY_TAG = "nightly";

export class Config {
    readonly extensionId = "matklad.rust-analyzer";

    private readonly rootSection = "rust-analyzer";
    private readonly requiresReloadOpts = [
        "serverPath",
        "cargoFeatures",
        "cargo-watch",
        "highlighting.semanticTokens",
        "inlayHints",
        "updates.channel",
    ]
        .map(opt => `${this.rootSection}.${opt}`);

    readonly package: {
        version: string;
        releaseTag: string | undefined;
        enableProposedApi: boolean | undefined;
    } = vscode.extensions.getExtension(this.extensionId)!.packageJSON;

    readonly globalStoragePath: string;

    constructor(ctx: vscode.ExtensionContext) {
        this.globalStoragePath = ctx.globalStoragePath;
        vscode.workspace.onDidChangeConfiguration(this.onDidChangeConfiguration, this, ctx.subscriptions);
        this.refreshLogging();
    }

    private refreshLogging() {
        log.setEnabled(this.traceExtension);
        log.debug(
            "Extension version:", this.package.version,
            "using configuration:", this.cfg
        );
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

    get serverPath() { return this.cfg.get<null | string>("serverPath")!; }
    get channel() { return this.cfg.get<UpdatesChannel>("updates.channel")!; }
    get askBeforeDownload() { return this.cfg.get<boolean>("updates.askBeforeDownload")!; }
    get highlightingSemanticTokens() { return this.cfg.get<boolean>("highlighting.semanticTokens")!; }
    get highlightingOn() { return this.cfg.get<boolean>("highlightingOn")!; }
    get rainbowHighlightingOn() { return this.cfg.get<boolean>("rainbowHighlightingOn")!; }
    get lruCapacity() { return this.cfg.get<null | number>("lruCapacity")!; }
    get excludeGlobs() { return this.cfg.get<string[]>("excludeGlobs")!; }
    get useClientWatching() { return this.cfg.get<boolean>("useClientWatching")!; }
    get featureFlags() { return this.cfg.get<Record<string, boolean>>("featureFlags")!; }
    get rustfmtArgs() { return this.cfg.get<string[]>("rustfmtArgs")!; }
    get loadOutDirsFromCheck() { return this.cfg.get<boolean>("loadOutDirsFromCheck")!; }
    get traceExtension() { return this.cfg.get<boolean>("trace.extension")!; }

    // for internal use
    get withSysroot() { return this.cfg.get<boolean>("withSysroot", true)!; }

    get inlayHints() {
        return {
            typeHints: this.cfg.get<boolean>("inlayHints.typeHints")!,
            parameterHints: this.cfg.get<boolean>("inlayHints.parameterHints")!,
            chainingHints: this.cfg.get<boolean>("inlayHints.chainingHints")!,
            maxLength: this.cfg.get<null | number>("inlayHints.maxLength")!,
        };
    }

    get cargoWatchOptions() {
        return {
            enable: this.cfg.get<boolean>("cargo-watch.enable")!,
            arguments: this.cfg.get<string[]>("cargo-watch.arguments")!,
            allTargets: this.cfg.get<boolean>("cargo-watch.allTargets")!,
            command: this.cfg.get<string>("cargo-watch.command")!,
        };
    }

    get cargoFeatures() {
        return {
            noDefaultFeatures: this.cfg.get<boolean>("cargoFeatures.noDefaultFeatures")!,
            allFeatures: this.cfg.get<boolean>("cargoFeatures.allFeatures")!,
            features: this.cfg.get<string[]>("cargoFeatures.features")!,
            loadOutDirsFromCheck: this.cfg.get<boolean>("cargoFeatures.loadOutDirsFromCheck")!,
        };
    }
}
