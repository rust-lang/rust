import * as vscode from 'vscode';

import { Server } from './server';

const RA_LSP_DEBUG = process.env.__RA_LSP_SERVER_DEBUG;

export class Config {
    public highlightingOn = true;
    public raLspServerPath = RA_LSP_DEBUG || 'ra_lsp_server';

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

        if (!this.highlightingOn && Server) {
            Server.highlighter.removeHighlights();
        }

        if (config.has('raLspServerPath')) {
            this.raLspServerPath =
                RA_LSP_DEBUG || (config.get('raLspServerPath') as string);
        }
    }
}
