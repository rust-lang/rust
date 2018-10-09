import * as vscode from 'vscode';

import { Decoration } from '../highlighting';
import { Server } from '../server';

export interface PublishDecorationsParams {
    uri: string;
    decorations: Decoration[];
}

export function handle(params: PublishDecorationsParams) {
    const targetEditor = vscode.window.visibleTextEditors.find(
        editor => editor.document.uri.toString() === params.uri
    );
    if (!Server.config.highlightingOn || !targetEditor) {
        return;
    }
    Server.highlighter.setHighlights(targetEditor, params.decorations);
}
