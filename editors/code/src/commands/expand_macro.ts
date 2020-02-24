import * as vscode from 'vscode';
import * as ra from '../rust-analyzer-api';

import { Ctx, Cmd } from '../ctx';

// Opens the virtual file that will show the syntax tree
//
// The contents of the file come from the `TextDocumentContentProvider`
export function expandMacro(ctx: Ctx): Cmd {
    const tdcp = new TextDocumentContentProvider(ctx);
    ctx.pushCleanup(
        vscode.workspace.registerTextDocumentContentProvider(
            'rust-analyzer',
            tdcp,
        ),
    );

    return async () => {
        const document = await vscode.workspace.openTextDocument(tdcp.uri);
        tdcp.eventEmitter.fire(tdcp.uri);
        return vscode.window.showTextDocument(
            document,
            vscode.ViewColumn.Two,
            true,
        );
    };
}

function codeFormat(expanded: ra.ExpandedMacro): string {
    let result = `// Recursive expansion of ${expanded.name}! macro\n`;
    result += '// ' + '='.repeat(result.length - 3);
    result += '\n\n';
    result += expanded.expansion;

    return result;
}

class TextDocumentContentProvider
    implements vscode.TextDocumentContentProvider {
    uri = vscode.Uri.parse('rust-analyzer://expandMacro/[EXPANSION].rs');
    eventEmitter = new vscode.EventEmitter<vscode.Uri>();

    constructor(private readonly ctx: Ctx) {
    }

    async provideTextDocumentContent(_uri: vscode.Uri): Promise<string> {
        const editor = vscode.window.activeTextEditor;
        const client = this.ctx.client;
        if (!editor || !client) return '';

        const position = editor.selection.active;

        const expanded = await client.sendRequest(ra.expandMacro, {
            textDocument: { uri: editor.document.uri.toString() },
            position,
        });

        if (expanded == null) return 'Not available';

        return codeFormat(expanded);
    }

    get onDidChange(): vscode.Event<vscode.Uri> {
        return this.eventEmitter.event;
    }
}
