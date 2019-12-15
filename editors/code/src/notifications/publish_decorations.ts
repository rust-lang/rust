import * as vscode from 'vscode';

import { Decoration } from '../highlighting';
import { Server } from '../server';

export interface PublishDecorationsParams {
    uri: string;
    decorations: Decoration[];
}

export function handle(params: PublishDecorationsParams) {
    const targetEditor = vscode.window.visibleTextEditors.find(
        editor => {
            const unescapedUri = unescape(editor.document.uri.toString());
            // Unescaped URI looks like:
            // file:///c:/Workspace/ra-test/src/main.rs
            return unescapedUri === params.uri
        }
    );

    if (!Server.config.highlightingOn || !targetEditor) {
        return;
    }

    Server.highlighter.setHighlights(targetEditor, params.decorations);
}
