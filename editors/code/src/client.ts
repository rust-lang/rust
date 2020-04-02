import * as lc from 'vscode-languageclient';
import * as vscode from 'vscode';

import { Config } from './config';
import { CallHierarchyFeature } from 'vscode-languageclient/lib/callHierarchy.proposed';
import { SemanticTokensFeature, DocumentSemanticsTokensSignature } from 'vscode-languageclient/lib/semanticTokens.proposed';

export async function createClient(config: Config, serverPath: string, cwd: string): Promise<lc.LanguageClient> {
    // '.' Is the fallback if no folder is open
    // TODO?: Workspace folders support Uri's (eg: file://test.txt).
    // It might be a good idea to test if the uri points to a file.

    const run: lc.Executable = {
        command: serverPath,
        options: { cwd },
    };
    const serverOptions: lc.ServerOptions = {
        run,
        debug: run,
    };
    const traceOutputChannel = vscode.window.createOutputChannel(
        'Rust Analyzer Language Server Trace',
    );

    const clientOptions: lc.LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'rust' }],
        initializationOptions: vscode.workspace.getConfiguration("rust-analyzer"),
        traceOutputChannel,
        middleware: {
            // Workaround for https://github.com/microsoft/vscode-languageserver-node/issues/576
            async provideDocumentSemanticTokens(document: vscode.TextDocument, token: vscode.CancellationToken, next: DocumentSemanticsTokensSignature) {
                const res = await next(document, token);
                if (res === undefined) throw new Error('busy');
                return res;
            }
        } as any
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

    // To turn on all proposed features use: res.registerProposedFeatures();
    // Here we want to just enable CallHierarchyFeature since it is available on stable.
    // Note that while the CallHierarchyFeature is stable the LSP protocol is not.
    res.registerFeature(new CallHierarchyFeature(res));

    if (config.package.enableProposedApi) {
        if (config.highlightingSemanticTokens) {
            res.registerFeature(new SemanticTokensFeature(res));
        }
    }

    return res;
}
