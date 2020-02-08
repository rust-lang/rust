import * as lc from 'vscode-languageclient';

import { window, workspace } from 'vscode';
import { Config } from './config';
import { ensureLanguageServerBinary } from './installation/language_server';

export async function createClient(config: Config): Promise<null | lc.LanguageClient> {
    // '.' Is the fallback if no folder is open
    // TODO?: Workspace folders support Uri's (eg: file://test.txt).
    // It might be a good idea to test if the uri points to a file.
    const workspaceFolderPath = workspace.workspaceFolders?.[0]?.uri.fsPath ?? '.';

    const raLspServerPath = await ensureLanguageServerBinary(config.langServerSource);
    if (!raLspServerPath) return null;

    const run: lc.Executable = {
        command: raLspServerPath,
        options: { cwd: workspaceFolderPath },
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
            lruCapacity: config.lruCapacity,
            maxInlayHintLength: config.maxInlayHintLength,
            cargoWatchEnable: config.cargoWatchOptions.enable,
            cargoWatchArgs: config.cargoWatchOptions.arguments,
            cargoWatchCommand: config.cargoWatchOptions.command,
            cargoWatchAllTargets: config.cargoWatchOptions.allTargets,
            excludeGlobs: config.excludeGlobs,
            useClientWatching: config.useClientWatching,
            featureFlags: config.featureFlags,
            withSysroot: config.withSysroot,
            cargoFeatures: config.cargoFeatures,
        },
        traceOutputChannel,
    };

    const res = new lc.LanguageClient(
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
    res._tracer = {
        log: (messageOrDataObject: string | unknown, data?: string) => {
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
                    res.logTrace(messageOrDataObject, data);
                }
            } else {
                // @ts-ignore
                res.logObjectTrace(messageOrDataObject);
            }
        },
    };
    res.registerProposedFeatures();
    return res;
}
