import * as os from "os";
import * as vscode from 'vscode';
import { BinarySource } from "./installation/interfaces";

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
    langServerSource!: null | BinarySource;

    highlightingOn = true;
    rainbowHighlightingOn = false;
    enableEnhancedTyping = true;
    lruCapacity: null | number = null;
    displayInlayHints = true;
    maxInlayHintLength: null | number = null;
    excludeGlobs: string[] = [];
    useClientWatching = true;
    featureFlags: Record<string, boolean> = {};
    // for internal use
    withSysroot: null | boolean = null;
    cargoWatchOptions: CargoWatchOptions = {
        enable: true,
        arguments: [],
        command: '',
        allTargets: true,
    };
    cargoFeatures: CargoFeatures = {
        noDefaultFeatures: false,
        allFeatures: true,
        features: [],
    };

    private prevEnhancedTyping: null | boolean = null;
    private prevCargoFeatures: null | CargoFeatures = null;
    private prevCargoWatchOptions: null | CargoWatchOptions = null;

    constructor(ctx: vscode.ExtensionContext) {
        vscode.workspace.onDidChangeConfiguration(_ => this.refresh(ctx), null, ctx.subscriptions);
        this.refresh(ctx);
    }

    private static expandPathResolving(path: string) {
        if (path.startsWith('~/')) {
            return path.replace('~', os.homedir());
        }
        return path;
    }

    /**
     * Name of the binary artifact for `ra_lsp_server` that is published for
     * `platform` on GitHub releases. (It is also stored under the same name when
     * downloaded by the extension).
     */
    private static prebuiltLangServerFileName(platform: NodeJS.Platform): null | string {
        switch (platform) {
            case "linux":  return "ra_lsp_server-linux";
            case "darwin": return "ra_lsp_server-mac";
            case "win32":  return "ra_lsp_server-windows.exe";

            // Users on these platforms yet need to manually build from sources
            case "aix":
            case "android":
            case "freebsd":
            case "openbsd":
            case "sunos":
            case "cygwin":
            case "netbsd": return null;
            // The list of platforms is exhaustive see (`NodeJS.Platform` type definition)
        }
    }

    private static langServerBinarySource(
        ctx: vscode.ExtensionContext,
        config: vscode.WorkspaceConfiguration
    ): null | BinarySource {
        const raLspServerPath = RA_LSP_DEBUG ?? config.get<null | string>("raLspServerPath");

        if (raLspServerPath) {
            return {
                type: BinarySource.Type.ExplicitPath,
                path: Config.expandPathResolving(raLspServerPath)
            };
        }

        const prebuiltBinaryName = Config.prebuiltLangServerFileName(process.platform);

        return !prebuiltBinaryName ? null : {
            type: BinarySource.Type.GithubRelease,
            dir: ctx.globalStoragePath,
            file: prebuiltBinaryName,
            repo: {
                name: "rust-analyzer",
                owner: "rust-analyzer",
            }
        };
    }


    // FIXME: revisit the logic for `if (.has(...)) config.get(...)` set default
    // values only in one place (i.e. remove default values from non-readonly members declarations)
    private refresh(ctx: vscode.ExtensionContext) {
        const config = vscode.workspace.getConfiguration('rust-analyzer');

        let requireReloadMessage = null;

        if (config.has('highlightingOn')) {
            this.highlightingOn = config.get('highlightingOn') as boolean;
        }

        if (config.has('rainbowHighlightingOn')) {
            this.rainbowHighlightingOn = config.get(
                'rainbowHighlightingOn',
            ) as boolean;
        }

        if (config.has('enableEnhancedTyping')) {
            this.enableEnhancedTyping = config.get(
                'enableEnhancedTyping',
            ) as boolean;

            if (this.prevEnhancedTyping === null) {
                this.prevEnhancedTyping = this.enableEnhancedTyping;
            }
        } else if (this.prevEnhancedTyping === null) {
            this.prevEnhancedTyping = this.enableEnhancedTyping;
        }

        if (this.prevEnhancedTyping !== this.enableEnhancedTyping) {
            requireReloadMessage =
                'Changing enhanced typing setting requires a reload';
            this.prevEnhancedTyping = this.enableEnhancedTyping;
        }

        this.langServerSource = Config.langServerBinarySource(ctx, config);

        if (config.has('cargo-watch.enable')) {
            this.cargoWatchOptions.enable = config.get<boolean>(
                'cargo-watch.enable',
                true,
            );
        }

        if (config.has('cargo-watch.arguments')) {
            this.cargoWatchOptions.arguments = config.get<string[]>(
                'cargo-watch.arguments',
                [],
            );
        }

        if (config.has('cargo-watch.command')) {
            this.cargoWatchOptions.command = config.get<string>(
                'cargo-watch.command',
                '',
            );
        }

        if (config.has('cargo-watch.allTargets')) {
            this.cargoWatchOptions.allTargets = config.get<boolean>(
                'cargo-watch.allTargets',
                true,
            );
        }

        if (config.has('lruCapacity')) {
            this.lruCapacity = config.get('lruCapacity') as number;
        }

        if (config.has('displayInlayHints')) {
            this.displayInlayHints = config.get('displayInlayHints') as boolean;
        }
        if (config.has('maxInlayHintLength')) {
            this.maxInlayHintLength = config.get(
                'maxInlayHintLength',
            ) as number;
        }
        if (config.has('excludeGlobs')) {
            this.excludeGlobs = config.get('excludeGlobs') || [];
        }
        if (config.has('useClientWatching')) {
            this.useClientWatching = config.get('useClientWatching') || true;
        }
        if (config.has('featureFlags')) {
            this.featureFlags = config.get('featureFlags') || {};
        }
        if (config.has('withSysroot')) {
            this.withSysroot = config.get('withSysroot') || false;
        }

        if (config.has('cargoFeatures.noDefaultFeatures')) {
            this.cargoFeatures.noDefaultFeatures = config.get(
                'cargoFeatures.noDefaultFeatures',
                false,
            );
        }
        if (config.has('cargoFeatures.allFeatures')) {
            this.cargoFeatures.allFeatures = config.get(
                'cargoFeatures.allFeatures',
                true,
            );
        }
        if (config.has('cargoFeatures.features')) {
            this.cargoFeatures.features = config.get(
                'cargoFeatures.features',
                [],
            );
        }

        if (
            this.prevCargoFeatures !== null &&
            (this.cargoFeatures.allFeatures !==
                this.prevCargoFeatures.allFeatures ||
                this.cargoFeatures.noDefaultFeatures !==
                this.prevCargoFeatures.noDefaultFeatures ||
                this.cargoFeatures.features.length !==
                this.prevCargoFeatures.features.length ||
                this.cargoFeatures.features.some(
                    (v, i) => v !== this.prevCargoFeatures!.features[i],
                ))
        ) {
            requireReloadMessage = 'Changing cargo features requires a reload';
        }
        this.prevCargoFeatures = { ...this.cargoFeatures };

        if (this.prevCargoWatchOptions !== null) {
            const changed =
                this.cargoWatchOptions.enable !== this.prevCargoWatchOptions.enable ||
                this.cargoWatchOptions.command !== this.prevCargoWatchOptions.command ||
                this.cargoWatchOptions.allTargets !== this.prevCargoWatchOptions.allTargets ||
                this.cargoWatchOptions.arguments.length !== this.prevCargoWatchOptions.arguments.length ||
                this.cargoWatchOptions.arguments.some(
                    (v, i) => v !== this.prevCargoWatchOptions!.arguments[i],
                );
            if (changed) {
                requireReloadMessage = 'Changing cargo-watch options requires a reload';
            }
        }
        this.prevCargoWatchOptions = { ...this.cargoWatchOptions };

        if (requireReloadMessage !== null) {
            const reloadAction = 'Reload now';
            vscode.window
                .showInformationMessage(requireReloadMessage, reloadAction)
                .then(selectedAction => {
                    if (selectedAction === reloadAction) {
                        vscode.commands.executeCommand(
                            'workbench.action.reloadWindow',
                        );
                    }
                });
        }
    }
}
