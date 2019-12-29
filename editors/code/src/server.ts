import { lookpath } from 'lookpath';
import { homedir, platform } from 'os';
import * as lc from 'vscode-languageclient';

import { window, workspace } from 'vscode';
import { Config } from './config';
import { Highlighter } from './highlighting';

function expandPathResolving(path: string) {
    if (path.startsWith('~/')) {
        return path.replace('~', homedir());
    }
    return path;
}

export class Server {
    public static highlighter = new Highlighter();
    public static config = new Config();
    public static client: lc.LanguageClient;

    public static async start(
        notificationHandlers: Iterable<[string, lc.GenericNotificationHandler]>,
    ) {
        // '.' Is the fallback if no folder is open
        // TODO?: Workspace folders support Uri's (eg: file://test.txt). It might be a good idea to test if the uri points to a file.
        let folder: string = '.';
        if (workspace.workspaceFolders !== undefined) {
            folder = workspace.workspaceFolders[0].uri.fsPath.toString();
        }

        const command = expandPathResolving(this.config.raLspServerPath);
        // FIXME: remove check when the following issue is fixed:
        // https://github.com/otiai10/lookpath/issues/4
        if (platform() !== 'win32') {
            if (!(await lookpath(command))) {
                throw new Error(
                    `Cannot find rust-analyzer server \`${command}\` in PATH.`,
                );
            }
        }
        const run: lc.Executable = {
            command,
            options: { cwd: folder },
        };
        const serverOptions: lc.ServerOptions = {
            run,
            debug: run,
        };
        const traceOutputChannel = window.createOutputChannel(
            'Rust Analyzer Language Server Trace',
        );
        const clientOptions: lc.LanguageClientOptions = {
            documentSelector: [{ scheme: 'file', language: 'rust' }],
            initializationOptions: {
                publishDecorations: true,
                lruCapacity: Server.config.lruCapacity,
                maxInlayHintLength: Server.config.maxInlayHintLength,
                cargoWatchEnable: Server.config.cargoWatchOptions.enable,
                cargoWatchArgumets: Server.config.cargoWatchOptions.arguments,
                cargoWatchCommand: Server.config.cargoWatchOptions.command,
                cargoWatchAllTargets:
                    Server.config.cargoWatchOptions.allTargets,
                excludeGlobs: Server.config.excludeGlobs,
                useClientWatching: Server.config.useClientWatching,
                featureFlags: Server.config.featureFlags,
                withSysroot: Server.config.withSysroot,
                cargoFeatures: Server.config.cargoFeatures,
            },
            traceOutputChannel,
        };

        Server.client = new lc.LanguageClient(
            'rust-analyzer',
            'Rust Analyzer Language Server',
            serverOptions,
            clientOptions,
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
                            'rust-analyzer/publishDecorations',
                        ) ||
                        messageOrDataObject.includes(
                            'rust-analyzer/decorationsRequest',
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
            },
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
