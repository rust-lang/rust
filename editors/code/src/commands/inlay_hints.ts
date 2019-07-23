import * as vscode from 'vscode';
import { DecorationOptions, Range, TextDocumentChangeEvent, TextDocumentContentChangeEvent, TextEditor } from 'vscode';
import { TextDocumentIdentifier } from 'vscode-languageclient';
import { Server } from '../server';

interface InlayHintsParams {
    textDocument: TextDocumentIdentifier;
}

interface InlayHint {
    range: Range,
    kind: string,
    label: string,
}

const typeHintDecorationType = vscode.window.createTextEditorDecorationType({
    after: {
        color: new vscode.ThemeColor('ralsp.inlayHint'),
    },
});

export class HintsUpdater {
    private currentDecorations = new Map<string, DecorationOptions[]>();
    private displayHints = true;

    public async loadHints(editor: vscode.TextEditor | undefined): Promise<void> {
        if (this.displayHints && editor !== undefined) {
            await this.updateDecorationsFromServer(editor.document.uri.toString(), editor);
        }
    }

    public dropHints(document: vscode.TextDocument) {
        if (this.displayHints) {
            this.currentDecorations.delete(document.uri.toString());
        }
    }

    public async toggleHintsDisplay(displayHints: boolean): Promise<void> {
        if (this.displayHints !== displayHints) {
            this.displayHints = displayHints;
            this.currentDecorations.clear();

            if (displayHints) {
                return this.updateHints();
            } else {
                const editor = vscode.window.activeTextEditor;
                if (editor != null) {
                    return editor.setDecorations(typeHintDecorationType, [])
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
        if (document.languageId !== 'rust') {
            return;
        }

        const documentUri = document.uri.toString();
        const documentDecorators = this.currentDecorations.get(documentUri) || [];

        if (documentDecorators.length > 0) {
            // FIXME a dbg! in the handlers.rs of the server causes
            // an endless storm of events with `cause.contentChanges` with the dbg messages, why?
            const changesFromFile = cause !== undefined ? cause.contentChanges.filter(changeEvent => this.isEventInFile(document.lineCount, changeEvent)) : [];
            if (changesFromFile.length === 0) {
                return;
            }

            const firstShiftedLine = this.getFirstShiftedLine(changesFromFile);
            if (firstShiftedLine !== null) {
                const unchangedDecorations = documentDecorators.filter(decoration => decoration.range.start.line < firstShiftedLine);
                if (unchangedDecorations.length !== documentDecorators.length) {
                    await editor.setDecorations(typeHintDecorationType, unchangedDecorations);
                }
            }
        }
        return await this.updateDecorationsFromServer(documentUri, editor);
    }

    private isEventInFile(documentLineCount: number, event: TextDocumentContentChangeEvent): boolean {
        const eventText = event.text;
        if (eventText.length === 0) {
            return event.range.start.line <= documentLineCount || event.range.end.line <= documentLineCount;
        } else {
            return event.range.start.line <= documentLineCount && event.range.end.line <= documentLineCount;
        }
    }

    private getFirstShiftedLine(changeEvents: TextDocumentContentChangeEvent[]): number | null {
        let topmostUnshiftedLine: number | null = null;

        changeEvents
            .filter(event => this.isShiftingChange(event))
            .forEach(event => {
                const shiftedLineNumber = event.range.start.line;
                if (topmostUnshiftedLine === null || topmostUnshiftedLine > shiftedLineNumber) {
                    topmostUnshiftedLine = shiftedLineNumber;
                }
            });

        return topmostUnshiftedLine;
    }

    private isShiftingChange(event: TextDocumentContentChangeEvent) {
        const eventText = event.text;
        if (eventText.length === 0) {
            return !event.range.isSingleLine;
        } else {
            return eventText.indexOf('\n') >= 0 || eventText.indexOf('\r') >= 0;
        }
    }

    private async updateDecorationsFromServer(documentUri: string, editor: TextEditor): Promise<void> {
        const newHints = await this.queryHints(documentUri) || [];
        const newDecorations = newHints.map(hint => (
            {
                range: hint.range,
                renderOptions: { after: { contentText: `: ${hint.label}` } },
            }
        ));
        this.currentDecorations.set(documentUri, newDecorations);
        return editor.setDecorations(typeHintDecorationType, newDecorations);
    }

    private async queryHints(documentUri: string): Promise<InlayHint[] | null> {
        const request: InlayHintsParams = { textDocument: { uri: documentUri } };
        const client = Server.client;
        return client.onReady().then(() => client.sendRequest<InlayHint[] | null>(
            'rust-analyzer/inlayHints',
            request
        ));
    }
}
