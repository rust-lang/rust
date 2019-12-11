import * as vscode from 'vscode';
import { Server } from '../server';

const statusUri = vscode.Uri.parse('rust-analyzer-status://status');

export class TextDocumentContentProvider
    implements vscode.TextDocumentContentProvider {
    public eventEmitter = new vscode.EventEmitter<vscode.Uri>();
    public syntaxTree: string = 'Not available';

    public provideTextDocumentContent(
        _uri: vscode.Uri,
    ): vscode.ProviderResult<string> {
        const editor = vscode.window.activeTextEditor;
        if (editor == null) {
            return '';
        }
        return Server.client.sendRequest<string>(
            'rust-analyzer/analyzerStatus',
            null,
        );
    }

    get onDidChange(): vscode.Event<vscode.Uri> {
        return this.eventEmitter.event;
    }
}

let poller: NodeJS.Timer | null = null;

// Shows status of rust-analyzer (for debugging)

export function makeCommand(context: vscode.ExtensionContext) {
    const textDocumentContentProvider = new TextDocumentContentProvider();
    context.subscriptions.push(
        vscode.workspace.registerTextDocumentContentProvider(
            'rust-analyzer-status',
            textDocumentContentProvider,
        ),
    );

    context.subscriptions.push({
        dispose() {
            if (poller != null) {
                clearInterval(poller);
            }
        },
    });

    return async function handle() {
        if (poller == null) {
            poller = setInterval(
                () => textDocumentContentProvider.eventEmitter.fire(statusUri),
                1000,
            );
        }
        const document = await vscode.workspace.openTextDocument(statusUri);
        return vscode.window.showTextDocument(
            document,
            vscode.ViewColumn.Two,
            true,
        );
    };
}
