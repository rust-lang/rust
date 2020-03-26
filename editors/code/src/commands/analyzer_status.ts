import * as vscode from 'vscode';

import * as ra from '../rust-analyzer-api';
import { Ctx, Cmd } from '../ctx';

// Shows status of rust-analyzer (for debugging)
export function analyzerStatus(ctx: Ctx): Cmd {
    let poller: NodeJS.Timer | undefined = undefined;
    const tdcp = new TextDocumentContentProvider(ctx);

    ctx.pushCleanup(
        vscode.workspace.registerTextDocumentContentProvider(
            'rust-analyzer-status',
            tdcp,
        ),
    );

    ctx.pushCleanup({
        dispose() {
            if (poller !== undefined) {
                clearInterval(poller);
            }
        },
    });

    return async () => {
        if (poller === undefined) {
            poller = setInterval(() => tdcp.eventEmitter.fire(tdcp.uri), 1000);
        }
        const document = await vscode.workspace.openTextDocument(tdcp.uri);
        return vscode.window.showTextDocument(document, vscode.ViewColumn.Two, true);
    };
}

class TextDocumentContentProvider implements vscode.TextDocumentContentProvider {
    readonly uri = vscode.Uri.parse('rust-analyzer-status://status');
    readonly eventEmitter = new vscode.EventEmitter<vscode.Uri>();

    constructor(private readonly ctx: Ctx) {
    }

    provideTextDocumentContent(_uri: vscode.Uri): vscode.ProviderResult<string> {
        if (!vscode.window.activeTextEditor) return '';

        return this.ctx.client.sendRequest(ra.analyzerStatus, null);
    }

    get onDidChange(): vscode.Event<vscode.Uri> {
        return this.eventEmitter.event;
    }
}
