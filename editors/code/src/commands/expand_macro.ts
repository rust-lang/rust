import * as vscode from 'vscode';
import { Position, TextDocumentIdentifier } from 'vscode-languageclient';
import { Server } from '../server';

export const expandMacroUri = vscode.Uri.parse(
    'rust-analyzer://expandMacro/[EXPANSION].rs'
);

export class ExpandMacroContentProvider
    implements vscode.TextDocumentContentProvider {
    public eventEmitter = new vscode.EventEmitter<vscode.Uri>();

    public provideTextDocumentContent(
        uri: vscode.Uri
    ): vscode.ProviderResult<string> {
        async function handle() {
            const editor = vscode.window.activeTextEditor;
            if (editor == null) {
                return '';
            }

            const position = editor.selection.active;
            const request: MacroExpandParams = {
                textDocument: { uri: editor.document.uri.toString() },
                position
            };
            const expanded = await Server.client.sendRequest<ExpandedMacro>(
                'rust-analyzer/expandMacro',
                request
            );

            if (expanded == null) {
                return 'Not available';
            }

            return code_format(expanded);
        }

        return handle();
    }

    get onDidChange(): vscode.Event<vscode.Uri> {
        return this.eventEmitter.event;
    }
}

// Opens the virtual file that will show the syntax tree
//
// The contents of the file come from the `TextDocumentContentProvider`
export function createHandle(provider: ExpandMacroContentProvider) {
    return async () => {
        const uri = expandMacroUri;

        const document = await vscode.workspace.openTextDocument(uri);

        provider.eventEmitter.fire(uri);

        return vscode.window.showTextDocument(
            document,
            vscode.ViewColumn.Two,
            true
        );
    };
}

interface MacroExpandParams {
    textDocument: TextDocumentIdentifier;
    position: Position;
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
