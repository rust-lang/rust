import * as vscode from 'vscode';

import { Ctx, Cmd } from '../ctx';

// Shows status of rust-analyzer (for debugging)
export function analyzerStatus(ctx: Ctx): Cmd {
    let poller: NodeJS.Timer | null = null;
    const tdcp = new TextDocumentContentProvider(ctx);

    ctx.pushCleanup(
        vscode.workspace.registerTextDocumentContentProvider(
            'rust-analyzer-status',
            tdcp,
        ),
    );

    ctx.pushCleanup({
        dispose() {
            if (poller != null) {
                clearInterval(poller);
            }
        },
    });

    return async function handle() {
        if (poller == null) {
            poller = setInterval(() => tdcp.eventEmitter.fire(tdcp.uri), 1000);
        }
        const document = await vscode.workspace.openTextDocument(tdcp.uri);
        return vscode.window.showTextDocument(
            document,
            vscode.ViewColumn.Two,
            true,
        );
    };
}

class TextDocumentContentProvider
    implements vscode.TextDocumentContentProvider {
    private ctx: Ctx;
    uri = vscode.Uri.parse('rust-analyzer-status://status');
    eventEmitter = new vscode.EventEmitter<vscode.Uri>();

    constructor(ctx: Ctx) {
        this.ctx = ctx;
    }

    provideTextDocumentContent(
        _uri: vscode.Uri,
    ): vscode.ProviderResult<string> {
        const editor = vscode.window.activeTextEditor;
        const client = this.ctx.client;
        if (!editor || !client) return '';

        return client.sendRequest<string>(
            'rust-analyzer/analyzerStatus',
            null,
        );
    }

    get onDidChange(): vscode.Event<vscode.Uri> {
        return this.eventEmitter.event;
    }
}
