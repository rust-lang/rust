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
            },
            async provideCodeActions(document: vscode.TextDocument, range: vscode.Range, context: vscode.CodeActionContext, token: vscode.CancellationToken, _next: lc.ProvideCodeActionsSignature) {
                const params: lc.CodeActionParams = {
                    textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(document),
                    range: client.code2ProtocolConverter.asRange(range),
                    context: client.code2ProtocolConverter.asCodeActionContext(context)
                };
                return client.sendRequest(lc.CodeActionRequest.type, params, token).then((values) => {
                    if (values === null) return undefined;
                    const result: (vscode.CodeAction | vscode.Command)[] = [];
                    for (const item of values) {
                        if (lc.CodeAction.is(item)) {
                            const action = client.protocol2CodeConverter.asCodeAction(item);
                            if (isSnippetEdit(item)) {
                                action.command = {
                                    command: "rust-analyzer.applySnippetWorkspaceEdit",
                                    title: "",
                                    arguments: [action.edit],
                                };
                                action.edit = undefined;
                            }
                            result.push(action);
                        } else {
                            const command = client.protocol2CodeConverter.asCommand(item);
                            result.push(command);
                        }
                    }
                    return result;
                },
                    (_error) => undefined
                );
            }

        } as any
    };

    const client = new lc.LanguageClient(
        'rust-analyzer',
        'Rust Analyzer Language Server',
        serverOptions,
        clientOptions,
    );

    // To turn on all proposed features use: client.registerProposedFeatures();
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
        capabilities.experimental = caps;
    }
    initialize(_capabilities: lc.ServerCapabilities<any>, _documentSelector: lc.DocumentSelector | undefined): void {
    }
}

function isSnippetEdit(action: lc.CodeAction): boolean {
    const documentChanges = action.edit?.documentChanges ?? [];
    for (const edit of documentChanges) {
        if (lc.TextDocumentEdit.is(edit)) {
            if (edit.edits.some((indel) => (indel as any).insertTextFormat === lc.InsertTextFormat.Snippet)) {
                return true;
            }
        }
    }
    return false;
}
