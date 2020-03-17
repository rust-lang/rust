import * as vscode from 'vscode';
import { log } from "./util";

export interface InlayHintOptions {
    typeHints: boolean;
    parameterHints: boolean;
    maxLength: number | null;
}

export interface CargoWatchOptions {
    enable: boolean;
    arguments: string[];
    command: string;
    allTargets: boolean;
}

export interface CargoFeatures {
    noDefaultFeatures: boolean;
    allFeatures: boolean;
    features: string[];
    loadOutDirsFromCheck: boolean;
}

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

    readonly packageJsonVersion = vscode
        .extensions
        .getExtension(this.extensionId)!
        .packageJSON
        .version as string; // n.n.YYYYMMDD[-nightly]

    /**
     * Either `nightly` or `YYYY-MM-DD` (i.e. `stable` release)
     */
    readonly extensionReleaseTag: string = (() => {
        if (this.packageJsonVersion.endsWith(NIGHTLY_TAG)) return NIGHTLY_TAG;

        const realVersionRegexp = /^\d+\.\d+\.(\d{4})(\d{2})(\d{2})/;
        const [, yyyy, mm, dd] = this.packageJsonVersion.match(realVersionRegexp)!;

        return `${yyyy}-${mm}-${dd}`;
    })();

    private cfg!: vscode.WorkspaceConfiguration;

    constructor(private readonly ctx: vscode.ExtensionContext) {
        vscode.workspace.onDidChangeConfiguration(this.onConfigChange, this, ctx.subscriptions);
        this.refreshConfig();
    }

    private refreshConfig() {
        this.cfg = vscode.workspace.getConfiguration(this.rootSection);
        const enableLogging = this.cfg.get("trace.extension") as boolean;
        log.setEnabled(enableLogging);
        log.debug(
            "Extension version:", this.packageJsonVersion,
            "using configuration:", this.cfg
        );
    }

    private async onConfigChange(event: vscode.ConfigurationChangeEvent) {
        this.refreshConfig();

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

    get globalStoragePath(): string { return this.ctx.globalStoragePath; }

    // We don't do runtime config validation here for simplicity. More on stackoverflow:
    // https://stackoverflow.com/questions/60135780/what-is-the-best-way-to-type-check-the-configuration-for-vscode-extension

    get serverPath() { return this.cfg.get("serverPath") as null | string; }
    get channel() { return this.cfg.get<"stable" | "nightly">("updates.channel")!; }
    get askBeforeDownload() { return this.cfg.get("updates.askBeforeDownload") as boolean; }
    get highlightingSemanticTokens() { return this.cfg.get("highlighting.semanticTokens") as boolean; }
    get highlightingOn() { return this.cfg.get("highlightingOn") as boolean; }
    get rainbowHighlightingOn() { return this.cfg.get("rainbowHighlightingOn") as boolean; }
    get lruCapacity() { return this.cfg.get("lruCapacity") as null | number; }
    get inlayHints(): InlayHintOptions {
        return {
            typeHints: this.cfg.get("inlayHints.typeHints") as boolean,
            parameterHints: this.cfg.get("inlayHints.parameterHints") as boolean,
            maxLength: this.cfg.get("inlayHints.maxLength") as null | number,
        };
    }
    get excludeGlobs() { return this.cfg.get("excludeGlobs") as string[]; }
    get useClientWatching() { return this.cfg.get("useClientWatching") as boolean; }
    get featureFlags() { return this.cfg.get("featureFlags") as Record<string, boolean>; }
    get rustfmtArgs() { return this.cfg.get("rustfmtArgs") as string[]; }
    get loadOutDirsFromCheck() { return this.cfg.get("loadOutDirsFromCheck") as boolean; }

    get cargoWatchOptions(): CargoWatchOptions {
        return {
            enable: this.cfg.get("cargo-watch.enable") as boolean,
            arguments: this.cfg.get("cargo-watch.arguments") as string[],
            allTargets: this.cfg.get("cargo-watch.allTargets") as boolean,
            command: this.cfg.get("cargo-watch.command") as string,
        };
    }

    get cargoFeatures(): CargoFeatures {
        return {
            noDefaultFeatures: this.cfg.get("cargoFeatures.noDefaultFeatures") as boolean,
            allFeatures: this.cfg.get("cargoFeatures.allFeatures") as boolean,
            features: this.cfg.get("cargoFeatures.features") as string[],
            loadOutDirsFromCheck: this.cfg.get("cargoFeatures.loadOutDirsFromCheck") as boolean,
        };
    }

    // for internal use
    get withSysroot() { return this.cfg.get("withSysroot", true) as boolean; }
}
