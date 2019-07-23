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
}

export class Config {
    public highlightingOn = true;
    public rainbowHighlightingOn = false;
    public enableEnhancedTyping = true;
    public raLspServerPath = RA_LSP_DEBUG || 'ra_lsp_server';
    public showWorkspaceLoadedNotification = true;
    public lruCapacity: null | number = null;
    public displayInlayHints = true;
    public cargoWatchOptions: CargoWatchOptions = {
        enableOnStartup: 'ask',
        trace: 'off',
        arguments: '',
        command: ''
    };

    private prevEnhancedTyping: null | boolean = null;

    constructor() {
        vscode.workspace.onDidChangeConfiguration(_ =>
            this.userConfigChanged()
        );
        this.userConfigChanged();
    }

    public userConfigChanged() {
        const config = vscode.workspace.getConfiguration('rust-analyzer');
        if (config.has('highlightingOn')) {
            this.highlightingOn = config.get('highlightingOn') as boolean;
        }

        if (config.has('rainbowHighlightingOn')) {
            this.rainbowHighlightingOn = config.get(
                'rainbowHighlightingOn'
            ) as boolean;
        }

        if (config.has('showWorkspaceLoadedNotification')) {
            this.showWorkspaceLoadedNotification = config.get(
                'showWorkspaceLoadedNotification'
            ) as boolean;
        }

        if (!this.highlightingOn && Server) {
            Server.highlighter.removeHighlights();
        }

        if (config.has('enableEnhancedTyping')) {
            this.enableEnhancedTyping = config.get(
                'enableEnhancedTyping'
            ) as boolean;

            if (this.prevEnhancedTyping === null) {
                this.prevEnhancedTyping = this.enableEnhancedTyping;
            }
        } else if (this.prevEnhancedTyping === null) {
            this.prevEnhancedTyping = this.enableEnhancedTyping;
        }

        if (this.prevEnhancedTyping !== this.enableEnhancedTyping) {
            const reloadAction = 'Reload now';
            vscode.window
                .showInformationMessage(
                    'Changing enhanced typing setting requires a reload',
                    reloadAction
                )
                .then(selectedAction => {
                    if (selectedAction === reloadAction) {
                        vscode.commands.executeCommand(
                            'workbench.action.reloadWindow'
                        );
                    }
                });
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
                'off'
            );
        }

        if (config.has('cargo-watch.arguments')) {
            this.cargoWatchOptions.arguments = config.get<string>(
                'cargo-watch.arguments',
                ''
            );
        }

        if (config.has('cargo-watch.command')) {
            this.cargoWatchOptions.command = config.get<string>(
                'cargo-watch.command',
                ''
            );
        }

        if (config.has('lruCapacity')) {
            this.lruCapacity = config.get('lruCapacity') as number;
        }

        if (config.has('displayInlayHints')) {
            this.displayInlayHints = config.get('displayInlayHints') as boolean;
        }
    }
}
