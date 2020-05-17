import * as lc from 'vscode-languageclient';
import * as vscode from 'vscode';

import { CallHierarchyFeature } from 'vscode-languageclient/lib/callHierarchy.proposed';
import { SemanticTokensFeature, DocumentSemanticsTokensSignature } from 'vscode-languageclient/lib/semanticTokens.proposed';

export function createClient(serverPath: string, cwd: string): lc.LanguageClient {
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

    const client = new lc.LanguageClient(
        'rust-analyzer',
        'Rust Analyzer Language Server',
        serverOptions,
        clientOptions,
    );

    // To turn on all proposed features use: res.registerProposedFeatures();
    // Here we want to enable CallHierarchyFeature and SemanticTokensFeature
    // since they are available on stable.
    // Note that while these features are stable in vscode their LSP protocol
    // implementations are still in the "proposed" category for 3.16.
    client.registerFeature(new CallHierarchyFeature(client));
    client.registerFeature(new SemanticTokensFeature(client));
    client.registerFeature(new SnippetTextEditFeature());

    return client;
}

class SnippetTextEditFeature implements lc.StaticFeature {
    fillClientCapabilities(capabilities: lc.ClientCapabilities): void {
        const caps: any = capabilities.experimental ?? {};
        caps.snippetTextEdit = true;
        capabilities.experimental = caps
    }
    initialize(_capabilities: lc.ServerCapabilities<any>, _documentSelector: lc.DocumentSelector | undefined): void {
    }
}
