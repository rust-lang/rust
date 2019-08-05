import * as vscode from 'vscode';
import { Range, TextDocumentChangeEvent, TextEditor } from 'vscode';
import { TextDocumentIdentifier } from 'vscode-languageclient';
import { Server } from '../server';

interface InlayHintsParams {
    textDocument: TextDocumentIdentifier;
}

interface InlayHint {
    range: Range;
    kind: string;
    label: string;
}

const typeHintDecorationType = vscode.window.createTextEditorDecorationType({
    after: {
        color: new vscode.ThemeColor('ralsp.inlayHint')
    }
});

export class HintsUpdater {
    private displayHints = true;
    private decorationsSinceLastChange = new Map<
        string,
        vscode.DecorationOptions[]
    >();

    public async loadHints(editor?: vscode.TextEditor): Promise<void> {
        if (this.displayHints) {
            const documentUri = this.getEditorDocumentUri(editor);
            if (documentUri !== null) {
                const latestDecorations = this.decorationsSinceLastChange.get(
                    documentUri.toString()
                );
                if (latestDecorations === undefined) {
                    await this.updateDecorationsFromServer(
                        documentUri,
                        editor!
                    );
                } else {
                    await editor!.setDecorations(
                        typeHintDecorationType,
                        latestDecorations
                    );
                }
            }
        }
    }

    public async toggleHintsDisplay(displayHints: boolean): Promise<void> {
        if (this.displayHints !== displayHints) {
            this.displayHints = displayHints;
            this.decorationsSinceLastChange.clear();

            if (displayHints) {
                return this.updateHints();
            } else {
                const currentEditor = vscode.window.activeTextEditor;
                if (this.getEditorDocumentUri(currentEditor) !== null) {
                    return currentEditor!.setDecorations(
                        typeHintDecorationType,
                        []
                    );
                }
            }
        }
    }

    public async updateHints(cause?: TextDocumentChangeEvent): Promise<void> {
        if (!this.displayHints) {
            return;
        }
        const editor = vscode.window.activeTextEditor;
        if (editor === undefined) {
            return;
        }
        const document = cause === undefined ? editor.document : cause.document;
        if (!this.isRustDocument(document)) {
            return;
        }

        this.decorationsSinceLastChange.clear();
        return await this.updateDecorationsFromServer(document.uri, editor);
    }

    private isRustDocument(document: vscode.TextDocument): boolean {
        return document && document.languageId === 'rust';
    }

    private async updateDecorationsFromServer(
        documentUri: vscode.Uri,
        editor: TextEditor
    ): Promise<void> {
        const newHints = await this.queryHints(documentUri.toString());
        if (
            newHints !== null &&
            this.getEditorDocumentUri(vscode.window.activeTextEditor) ===
                documentUri
        ) {
            const newDecorations = newHints.map(hint => ({
                range: hint.range,
                renderOptions: { after: { contentText: `: ${hint.label}` } }
            }));
            this.decorationsSinceLastChange.set(
                documentUri.toString(),
                newDecorations
            );
            return editor.setDecorations(
                typeHintDecorationType,
                newDecorations
            );
        }
    }

    private async queryHints(documentUri: string): Promise<InlayHint[] | null> {
        const request: InlayHintsParams = {
            textDocument: { uri: documentUri }
        };
        const client = Server.client;
        return client
            .onReady()
            .then(() =>
                client.sendRequest<InlayHint[] | null>(
                    'rust-analyzer/inlayHints',
                    request
                )
            );
    }

    private getEditorDocumentUri(
        editor?: vscode.TextEditor
    ): vscode.Uri | null {
        if (editor && this.isRustDocument(editor.document)) {
            return editor.document.uri;
        }
        return null;
    }
}
