import * as vscode from 'vscode';

import { Server } from './server';

const RA_LSP_DEBUG = process.env.__RA_LSP_SERVER_DEBUG;

export type CargoWatchStartupOptions = 'ask' | 'enabled' | 'disabled';
export type CargoWatchTraceOptions = 'off' | 'error' | 'verbose';

export interface CargoWatchOptions {
    enableOnStartup: CargoWatchStartupOptions;
    arguments: string;
    command: string;
    trace: CargoWatchTraceOptions;
    ignore: string[];
    allTargets: boolean;
}

export interface CargoFeatures {
    noDefaultFeatures: boolean;
    allFeatures: boolean;
    features: string[];
}

export class Config {
    public highlightingOn = true;
    public rainbowHighlightingOn = false;
    public enableEnhancedTyping = true;
    public raLspServerPath = RA_LSP_DEBUG || 'ra_lsp_server';
    public lruCapacity: null | number = null;
    public displayInlayHints = true;
    public maxInlayHintLength: null | number = null;
    public excludeGlobs = [];
    public useClientWatching = true;
    public featureFlags = {};
    // for internal use
    public withSysroot: null | boolean = null;
    public cargoWatchOptions: CargoWatchOptions = {
        enableOnStartup: 'ask',
        trace: 'off',
        arguments: '',
        command: '',
        ignore: [],
        allTargets: true,
    };
    public cargoFeatures: CargoFeatures = {
        noDefaultFeatures: false,
        allFeatures: true,
        features: [],
    };

    private prevEnhancedTyping: null | boolean = null;
    private prevCargoFeatures: null | CargoFeatures = null;

    constructor() {
        vscode.workspace.onDidChangeConfiguration(_ =>
            this.userConfigChanged(),
        );
        this.userConfigChanged();
    }

    public userConfigChanged() {
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

        if (!this.highlightingOn && Server) {
            Server.highlighter.removeHighlights();
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

        if (config.has('raLspServerPath')) {
            this.raLspServerPath =
                RA_LSP_DEBUG || (config.get('raLspServerPath') as string);
        }

        if (config.has('enableCargoWatchOnStartup')) {
            this.cargoWatchOptions.enableOnStartup = config.get<
                CargoWatchStartupOptions
            >('enableCargoWatchOnStartup', 'ask');
        }

        if (config.has('trace.cargo-watch')) {
            this.cargoWatchOptions.trace = config.get<CargoWatchTraceOptions>(
                'trace.cargo-watch',
                'off',
            );
        }

        if (config.has('cargo-watch.arguments')) {
            this.cargoWatchOptions.arguments = config.get<string>(
                'cargo-watch.arguments',
                '',
            );
        }

        if (config.has('cargo-watch.command')) {
            this.cargoWatchOptions.command = config.get<string>(
                'cargo-watch.command',
                '',
            );
        }

        if (config.has('cargo-watch.ignore')) {
            this.cargoWatchOptions.ignore = config.get<string[]>(
                'cargo-watch.ignore',
                [],
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
