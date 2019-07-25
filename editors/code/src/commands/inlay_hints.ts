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

    public async loadHints(
        editor: vscode.TextEditor | undefined
    ): Promise<void> {
        if (
            this.displayHints &&
            editor !== undefined &&
            this.isRustDocument(editor.document)
        ) {
            await this.updateDecorationsFromServer(
                editor.document.uri.toString(),
                editor
            );
        }
    }

    public async toggleHintsDisplay(displayHints: boolean): Promise<void> {
        if (this.displayHints !== displayHints) {
            this.displayHints = displayHints;

            if (displayHints) {
                return this.updateHints();
            } else {
                const editor = vscode.window.activeTextEditor;
                if (editor != null) {
                    return editor.setDecorations(typeHintDecorationType, []);
                }
            }
        }
    }

    public async updateHints(cause?: TextDocumentChangeEvent): Promise<void> {
        if (!this.displayHints) {
            return;
        }
        const editor = vscode.window.activeTextEditor;
        if (editor == null) {
            return;
        }
        const document = cause == null ? editor.document : cause.document;
        if (!this.isRustDocument(document)) {
            return;
        }

        return await this.updateDecorationsFromServer(
            document.uri.toString(),
            editor
        );
    }

    private isRustDocument(document: vscode.TextDocument): boolean {
        return document && document.languageId === 'rust';
    }

    private async updateDecorationsFromServer(
        documentUri: string,
        editor: TextEditor
    ): Promise<void> {
        const newHints = (await this.queryHints(documentUri)) || [];
        const newDecorations = newHints.map(hint => ({
            range: hint.range,
            renderOptions: { after: { contentText: `: ${hint.label}` } }
        }));
        return editor.setDecorations(typeHintDecorationType, newDecorations);
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
}
