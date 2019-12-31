import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

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

interface ExpandedMacro {
    name: string;
    expansion: string;
}

function code_format(expanded: ExpandedMacro): string {
    let result = `// Recursive expansion of ${expanded.name}! macro\n`;
    result += '// ' + '='.repeat(result.length - 3);
    result += '\n\n';
    result += expanded.expansion;

    return result;
}

class TextDocumentContentProvider
    implements vscode.TextDocumentContentProvider {
    private ctx: Ctx;
    uri = vscode.Uri.parse('rust-analyzer://expandMacro/[EXPANSION].rs');
    eventEmitter = new vscode.EventEmitter<vscode.Uri>();

    constructor(ctx: Ctx) {
        this.ctx = ctx;
    }

    async provideTextDocumentContent(_uri: vscode.Uri): Promise<string> {
        const editor = vscode.window.activeTextEditor;
        const client = this.ctx.client;
        if (!editor || !client) return '';

        const position = editor.selection.active;
        const request: lc.TextDocumentPositionParams = {
            textDocument: { uri: editor.document.uri.toString() },
            position,
        };
        const expanded = await client.sendRequest<ExpandedMacro>(
            'rust-analyzer/expandMacro',
            request,
        );

        if (expanded == null) return 'Not available';

        return code_format(expanded);
    }

    get onDidChange(): vscode.Event<vscode.Uri> {
        return this.eventEmitter.event;
    }
}
