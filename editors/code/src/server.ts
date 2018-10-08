import * as lc from 'vscode-languageclient';

import { Config } from './config';
import { Highlighter } from './highlighting';

export class Server {
    public static highlighter = new Highlighter();
    public static config = new Config();
    public static client: lc.LanguageClient;

    public static start(notificationHandlers: Iterable<[string, lc.GenericNotificationHandler]>) {
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
            for (const [type, handler] of notificationHandlers) {
                Server.client.onNotification(type, handler);
            }
        });
        Server.client.start();
    }
}
