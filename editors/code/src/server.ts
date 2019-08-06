import * as lc from 'vscode-languageclient';

import { window, workspace } from 'vscode';
import { Config } from './config';
import { Highlighter } from './highlighting';

export class Server {
    public static highlighter = new Highlighter();
    public static config = new Config();
    public static client: lc.LanguageClient;

    public static start(
        notificationHandlers: Iterable<[string, lc.GenericNotificationHandler]>
    ) {
        // '.' Is the fallback if no folder is open
        // TODO?: Workspace folders support Uri's (eg: file://test.txt). It might be a good idea to test if the uri points to a file.
        let folder: string = '.';
        if (workspace.workspaceFolders !== undefined) {
            folder = workspace.workspaceFolders[0].uri.fsPath.toString();
        }

        const run: lc.Executable = {
            command: this.config.raLspServerPath,
            options: { cwd: folder }
        };
        const serverOptions: lc.ServerOptions = {
            run,
            debug: run
        };
        const traceOutputChannel = window.createOutputChannel(
            'Rust Analyzer Language Server Trace'
        );
        const clientOptions: lc.LanguageClientOptions = {
            documentSelector: [{ scheme: 'file', language: 'rust' }],
            initializationOptions: {
                publishDecorations: true,
                showWorkspaceLoaded:
                    Server.config.showWorkspaceLoadedNotification,
                lruCapacity: Server.config.lruCapacity,
                excludeGlobs: Server.config.excludeGlobs
            },
            traceOutputChannel
        };

        Server.client = new lc.LanguageClient(
            'rust-analyzer',
            'Rust Analyzer Language Server',
            serverOptions,
            clientOptions
        );
        // HACK: This is an awful way of filtering out the decorations notifications
        // However, pending proper support, this is the most effecitve approach
        // Proper support for this would entail a change to vscode-languageclient to allow not notifying on certain messages
        // Or the ability to disable the serverside component of highlighting (but this means that to do tracing we need to disable hihlighting)
        // This also requires considering our settings strategy, which is work which needs doing
        // @ts-ignore The tracer is private to vscode-languageclient, but we need access to it to not log publishDecorations requests
        Server.client._tracer = {
            log: (messageOrDataObject: string | any, data?: string) => {
                if (typeof messageOrDataObject === 'string') {
                    if (
                        messageOrDataObject.includes(
                            'rust-analyzer/publishDecorations'
                        ) ||
                        messageOrDataObject.includes(
                            'rust-analyzer/decorationsRequest'
                        )
                    ) {
                        // Don't log publish decorations requests
                    } else {
                        // @ts-ignore This is just a utility function
                        Server.client.logTrace(messageOrDataObject, data);
                    }
                } else {
                    // @ts-ignore
                    Server.client.logObjectTrace(messageOrDataObject);
                }
            }
        };
        Server.client.registerProposedFeatures();
        Server.client.onReady().then(() => {
            for (const [type, handler] of notificationHandlers) {
                Server.client.onNotification(type, handler);
            }
        });
        Server.client.start();
    }
}
