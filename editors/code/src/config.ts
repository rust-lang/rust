import * as os from "os";
import * as vscode from 'vscode';
import { BinarySource } from "./installation/interfaces";
import { log } from "./util";

const RA_LSP_DEBUG = process.env.__RA_LSP_SERVER_DEBUG;

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
export class Config {
    private static readonly rootSection = "rust-analyzer";
    private static readonly requiresReloadOpts = [
        "cargoFeatures",
        "cargo-watch",
    ]
        .map(opt => `${Config.rootSection}.${opt}`);

    private static readonly extensionVersion: string = (() => {
        const packageJsonVersion = vscode
            .extensions
            .getExtension("matklad.rust-analyzer")!
            .packageJSON
            .version as string; // n.n.YYYYMMDD

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
        this.cfg = vscode.workspace.getConfiguration(Config.rootSection);
        const enableLogging = this.cfg.get("trace.extension") as boolean;
        log.setEnabled(enableLogging);
        log.debug("Using configuration:", this.cfg);
    }

    private async onConfigChange(event: vscode.ConfigurationChangeEvent) {
        this.refreshConfig();

        const requiresReloadOpt = Config.requiresReloadOpts.find(
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

    get serverSource(): null | BinarySource {
        const serverPath = RA_LSP_DEBUG ?? this.cfg.get<null | string>("serverPath");

        if (serverPath) {
            return {
                type: BinarySource.Type.ExplicitPath,
                path: Config.replaceTildeWithHomeDir(serverPath)
            };
        }

        const prebuiltBinaryName = this.prebuiltServerFileName;

        if (!prebuiltBinaryName) return null;

        return {
            type: BinarySource.Type.GithubRelease,
            dir: this.ctx.globalStoragePath,
            file: prebuiltBinaryName,
            storage: this.ctx.globalState,
            version: Config.extensionVersion,
            repo: {
                name: "rust-analyzer",
                owner: "rust-analyzer",
            }
        };
    }

    // We don't do runtime config validation here for simplicity. More on stackoverflow:
    // https://stackoverflow.com/questions/60135780/what-is-the-best-way-to-type-check-the-configuration-for-vscode-extension

    get highlightingOn() { return this.cfg.get("highlightingOn") as boolean; }
    get rainbowHighlightingOn() { return this.cfg.get("rainbowHighlightingOn") as boolean; }
    get lruCapacity() { return this.cfg.get("lruCapacity") as null | number; }
    get displayInlayHints() { return this.cfg.get("displayInlayHints") as boolean; }
    get maxInlayHintLength() { return this.cfg.get("maxInlayHintLength") as number; }
    get excludeGlobs() { return this.cfg.get("excludeGlobs") as string[]; }
    get useClientWatching() { return this.cfg.get("useClientWatching") as boolean; }
    get featureFlags() { return this.cfg.get("featureFlags") as Record<string, boolean>; }
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
