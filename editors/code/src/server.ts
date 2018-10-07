import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient'

import { Highlighter, Decoration } from './highlighting';

export class Config {
    highlightingOn = true;

    constructor() {
        vscode.workspace.onDidChangeConfiguration(_ => this.userConfigChanged());
        this.userConfigChanged();
    }

    userConfigChanged() {
        let config = vscode.workspace.getConfiguration('ra-lsp');
        if (config.has('highlightingOn')) {
            this.highlightingOn = config.get('highlightingOn') as boolean;
        };

        if (!this.highlightingOn) {
            Server.highlighter.removeHighlights();
        }
    }
}

export class Server {
    static highlighter = new Highlighter();
    static config = new Config();
    static client: lc.LanguageClient;


    static start() {
        let run: lc.Executable = {
            command: "ra_lsp_server",
            options: { cwd: "." }
        }
        let serverOptions: lc.ServerOptions = {
            run,
            debug: run
        };

        let clientOptions: lc.LanguageClientOptions = {
            documentSelector: [{ scheme: 'file', language: 'rust' }],
        };

        Server.client = new lc.LanguageClient(
            'ra-lsp',
            'rust-analyzer languge server',
            serverOptions,
            clientOptions,
        );
        Server.client.onReady().then(() => {
            Server.client.onNotification(
                "m/publishDecorations",
                (params: PublishDecorationsParams) => {
                    let editor = vscode.window.visibleTextEditors.find(
                        (editor) => editor.document.uri.toString() == params.uri
                    )
                    if (!Server.config.highlightingOn || !editor) return;
                    Server.highlighter.setHighlights(
                        editor,
                        params.decorations,
                    )
                }
            )
        })
        Server.client.start();
    }
}

interface PublishDecorationsParams {
    uri: string,
    decorations: Decoration[],
}
