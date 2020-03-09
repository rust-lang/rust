import * as os from "os";
import * as vscode from 'vscode';
import { ArtifactSource } from "./installation/interfaces";
import { log } from "./util";

const RA_LSP_DEBUG = process.env.__RA_LSP_SERVER_DEBUG;

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
}

export const enum UpdatesChannel {
    Stable = "stable",
    Nightly = "nightly"
}

export const NIGHTLY_TAG = "nightly";
export class Config {
    readonly extensionId = "matklad.rust-analyzer";

    private readonly rootSection = "rust-analyzer";
    private readonly requiresReloadOpts = [
        "cargoFeatures",
        "cargo-watch",
        "highlighting.semanticTokens",
        "inlayHints",
    ]
        .map(opt => `${this.rootSection}.${opt}`);

    /**
     * Either `nightly` or `YYYY-MM-DD` (i.e. `stable` release)
     */
    private readonly extensionVersion: string = (() => {
        const packageJsonVersion = vscode
            .extensions
            .getExtension(this.extensionId)!
            .packageJSON
            .version as string;

        if (packageJsonVersion.endsWith(NIGHTLY_TAG)) return NIGHTLY_TAG;

        const realVersionRegexp = /^\d+\.\d+\.(\d{4})(\d{2})(\d{2})/;
        const [, yyyy, mm, dd] = packageJsonVersion.match(realVersionRegexp)!;

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
        log.debug("Using configuration:", this.cfg);
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
            vscode.commands.executeCommand("workbench.action.reloadWindow");
        }
    }

    private static replaceTildeWithHomeDir(path: string) {
        if (path.startsWith("~/")) {
            return os.homedir() + path.slice("~".length);
        }
        return path;
    }

    /**
     * Name of the binary artifact for `rust-analyzer` that is published for
     * `platform` on GitHub releases. (It is also stored under the same name when
     * downloaded by the extension).
     */
    get prebuiltServerFileName(): null | string {
        // See possible `arch` values here:
        // https://nodejs.org/api/process.html#process_process_arch

        switch (process.platform) {

            case "linux": {
                switch (process.arch) {
                    case "arm":
                    case "arm64": return null;

                    default: return "rust-analyzer-linux";
                }
            }

            case "darwin": return "rust-analyzer-mac";
            case "win32": return "rust-analyzer-windows.exe";

            // Users on these platforms yet need to manually build from sources
            case "aix":
            case "android":
            case "freebsd":
            case "openbsd":
            case "sunos":
            case "cygwin":
            case "netbsd": return null;
            // The list of platforms is exhaustive (see `NodeJS.Platform` type definition)
        }
    }

    get installedExtensionUpdateChannel() {
        if (this.serverPath !== null) return null;

        return this.extensionVersion === NIGHTLY_TAG
            ? UpdatesChannel.Nightly
            : UpdatesChannel.Stable;
    }

    get serverSource(): null | ArtifactSource {
        const serverPath = RA_LSP_DEBUG ?? this.serverPath;

        if (serverPath) {
            return {
                type: ArtifactSource.Type.ExplicitPath,
                path: Config.replaceTildeWithHomeDir(serverPath)
            };
        }

        const prebuiltBinaryName = this.prebuiltServerFileName;

        if (!prebuiltBinaryName) return null;

        return this.createGithubReleaseSource(
            prebuiltBinaryName,
            this.extensionVersion
        );
    }

    private createGithubReleaseSource(file: string, tag: string): ArtifactSource.GithubRelease {
        return {
            type: ArtifactSource.Type.GithubRelease,
            file,
            tag,
            dir: this.ctx.globalStoragePath,
            repo: {
                name: "rust-analyzer",
                owner: "rust-analyzer"
            }
        }
    }

    get nightlyVsixSource(): ArtifactSource.GithubRelease {
        return this.createGithubReleaseSource("rust-analyzer.vsix", NIGHTLY_TAG);
    }

    readonly installedNightlyExtensionReleaseDate = new DateStorage(
        "rust-analyzer-installed-nightly-extension-release-date",
        this.ctx.globalState
    );
    readonly serverReleaseDate = new DateStorage(
        "rust-analyzer-server-release-date",
        this.ctx.globalState
    );
    readonly serverReleaseTag = new StringStorage(
        "rust-analyzer-release-tag", this.ctx.globalState
    );

    // We don't do runtime config validation here for simplicity. More on stackoverflow:
    // https://stackoverflow.com/questions/60135780/what-is-the-best-way-to-type-check-the-configuration-for-vscode-extension

    private get serverPath() { return this.cfg.get("serverPath") as null | string; }
    get updatesChannel() { return this.cfg.get("updates.channel") as UpdatesChannel; }
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
    get additionalOutDirs() { return this.cfg.get("additionalOutDirs") as Record<string, string>; }
    get rustfmtArgs() { return this.cfg.get("rustfmtArgs") as string[]; }

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
        };
    }

    // for internal use
    get withSysroot() { return this.cfg.get("withSysroot", true) as boolean; }
}

export class StringStorage {
    constructor(
        private readonly key: string,
        private readonly storage: vscode.Memento
    ) {}

    get(): null | string {
        const tag = this.storage.get(this.key, null);
        log.debug(this.key, "==", tag);
        return tag;
    }
    async set(tag: string) {
        log.debug(this.key, "=", tag);
        await this.storage.update(this.key, tag);
    }
}
export class DateStorage {

    constructor(
        private readonly key: string,
        private readonly storage: vscode.Memento
    ) {}

    get(): null | Date {
        const date = this.storage.get(this.key, null);
        log.debug(this.key, "==", date);
        return date ? new Date(date) : null;
    }

    async set(date: null | Date) {
        log.debug(this.key, "=", date);
        await this.storage.update(this.key, date);
    }
}
