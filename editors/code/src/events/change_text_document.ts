import * as vscode from 'vscode';

import {
    syntaxTreeUri,
    TextDocumentContentProvider
} from '../commands/syntaxTree';

export function createHandler(
    textDocumentContentProvider: TextDocumentContentProvider
) {
    return (event: vscode.TextDocumentChangeEvent) => {
        const doc = event.document;
        if (doc.languageId !== 'rust') {
            return;
        }
        afterLs(() => {
            textDocumentContentProvider.eventEmitter.fire(syntaxTreeUri);
        });
    };
}

// We need to order this after LS updates, but there's no API for that.
// Hence, good old setTimeout.
function afterLs(f: () => any) {
    setTimeout(f, 10);
}
