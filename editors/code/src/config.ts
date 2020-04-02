import * as vscode from 'vscode';
import { log } from "./util";

export type UpdatesChannel = "stable" | "nightly";

export const NIGHTLY_TAG = "nightly";

export class Config {
    readonly extensionId = "matklad.rust-analyzer";

    private readonly rootSection = "rust-analyzer";
    private readonly requiresReloadOpts = [
        "serverPath",
        "cargo",
        "files",
        "highlighting",
        "updates.channel",
    ]
        .map(opt => `${this.rootSection}.${opt}`);

    readonly package: {
        version: string;
        releaseTag: string | null;
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
    get traceExtension() { return this.cfg.get<boolean>("trace.extension")!; }

    get inlayHints() {
        return {
            typeHints: this.cfg.get<boolean>("inlayHints.typeHints")!,
            parameterHints: this.cfg.get<boolean>("inlayHints.parameterHints")!,
            chainingHints: this.cfg.get<boolean>("inlayHints.chainingHints")!,
            maxLength: this.cfg.get<null | number>("inlayHints.maxLength")!,
        };
    }

    get checkOnSave() {
        return {
            command: this.cfg.get<string>("checkOnSave.command")!,
        };
    }
}
