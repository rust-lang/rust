import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import { Config } from './config';
import { Decoration, Highlighter } from './highlighting';

export class Server {
    public static highlighter = new Highlighter();
    public static config = new Config();
    public static client: lc.LanguageClient;

    public static start() {
        const run: lc.Executable = {
            command: 'ra_lsp_server',
            options: { cwd: '.' },
        };
        const serverOptions: lc.ServerOptions = {
            run,
            debug: run,
        };

        const clientOptions: lc.LanguageClientOptions = {
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
                'm/publishDecorations',
                (params: PublishDecorationsParams) => {
                    const targetEditor = vscode.window.visibleTextEditors.find(
                        (editor) => editor.document.uri.toString() == params.uri,
                    );
                    if (!Server.config.highlightingOn || !targetEditor) { return; }
                    Server.highlighter.setHighlights(
                        targetEditor,
                        params.decorations,
                    );
                },
            );
        });
        Server.client.start();
    }
}

interface PublishDecorationsParams {
    uri: string;
    decorations: Decoration[];
}
