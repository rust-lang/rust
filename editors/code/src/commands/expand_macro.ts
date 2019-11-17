import * as vscode from 'vscode';
import { Position, TextDocumentIdentifier } from 'vscode-languageclient';
import { Server } from '../server';

type ExpandMacroResult = [string, string]

function code_format([name, text]: [string, string]): vscode.MarkdownString {
    const markdown = new vscode.MarkdownString(`#### Recursive expansion of ${name}! macro`);
    markdown.appendCodeblock(text, 'rust');    
    return markdown;
}

export class ExpandMacroHoverProvider implements vscode.HoverProvider {
    public provideHover(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken,
    ): Thenable<vscode.Hover | null> | null {
        async function handle() {
            const request: MacroExpandParams = {
                textDocument: { uri: document.uri.toString() },
                position,
            };
            const result = await Server.client.sendRequest<ExpandMacroResult>(
                'rust-analyzer/expandMacro',
                request
            );
            if (result != null) {
                const formated = code_format(result);
                return new vscode.Hover(formated);
            }

            return null;
        };

        return handle();
    }
}


interface MacroExpandParams {
    textDocument: TextDocumentIdentifier;
    position: Position;
}

