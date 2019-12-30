import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';
import { Server } from './server';
import { Ctx } from './ctx';

export function activateInlayHints(ctx: Ctx) {
    const hintsUpdater = new HintsUpdater();
    hintsUpdater.refreshHintsForVisibleEditors().then(() => {
        // vscode may ignore top level hintsUpdater.refreshHintsForVisibleEditors()
        // so update the hints once when the focus changes to guarantee their presence
        let editorChangeDisposable: vscode.Disposable | null = null;
        editorChangeDisposable = vscode.window.onDidChangeActiveTextEditor(
            _ => {
                if (editorChangeDisposable !== null) {
                    editorChangeDisposable.dispose();
                }
                return hintsUpdater.refreshHintsForVisibleEditors();
            },
        );

        ctx.pushCleanup(
            vscode.window.onDidChangeVisibleTextEditors(_ =>
                hintsUpdater.refreshHintsForVisibleEditors(),
            ),
        );
        ctx.pushCleanup(
            vscode.workspace.onDidChangeTextDocument(e =>
                hintsUpdater.refreshHintsForVisibleEditors(e),
            ),
        );
        ctx.pushCleanup(
            vscode.workspace.onDidChangeConfiguration(_ =>
                hintsUpdater.toggleHintsDisplay(
                    Server.config.displayInlayHints,
                ),
            ),
        );
    });
}

interface InlayHintsParams {
    textDocument: lc.TextDocumentIdentifier;
}

interface InlayHint {
    range: vscode.Range;
    kind: string;
    label: string;
}

const typeHintDecorationType = vscode.window.createTextEditorDecorationType({
    after: {
        color: new vscode.ThemeColor('ralsp.inlayHint'),
    },
});

class HintsUpdater {
    private displayHints = true;

    public async toggleHintsDisplay(displayHints: boolean): Promise<void> {
        if (this.displayHints !== displayHints) {
            this.displayHints = displayHints;
            return this.refreshVisibleEditorsHints(
                displayHints ? undefined : [],
            );
        }
    }

    public async refreshHintsForVisibleEditors(
        cause?: vscode.TextDocumentChangeEvent,
    ): Promise<void> {
        if (!this.displayHints) return;

        if (
            cause !== undefined &&
            (cause.contentChanges.length === 0 ||
                !this.isRustDocument(cause.document))
        ) {
            return;
        }
        return this.refreshVisibleEditorsHints();
    }

    private async refreshVisibleEditorsHints(
        newDecorations?: vscode.DecorationOptions[],
    ) {
        const promises: Array<Promise<void>> = [];

        for (const rustEditor of vscode.window.visibleTextEditors.filter(
            editor => this.isRustDocument(editor.document),
        )) {
            if (newDecorations !== undefined) {
                promises.push(
                    Promise.resolve(
                        rustEditor.setDecorations(
                            typeHintDecorationType,
                            newDecorations,
                        ),
                    ),
                );
            } else {
                promises.push(this.updateDecorationsFromServer(rustEditor));
            }
        }

        for (const promise of promises) {
            await promise;
        }
    }

    private isRustDocument(document: vscode.TextDocument): boolean {
        return document && document.languageId === 'rust';
    }

    private async updateDecorationsFromServer(
        editor: vscode.TextEditor,
    ): Promise<void> {
        const newHints = await this.queryHints(editor.document.uri.toString());
        if (newHints !== null) {
            const newDecorations = newHints.map(hint => ({
                range: hint.range,
                renderOptions: {
                    after: {
                        contentText: `: ${hint.label}`,
                    },
                },
            }));
            return editor.setDecorations(
                typeHintDecorationType,
                newDecorations,
            );
        }
    }

    private async queryHints(documentUri: string): Promise<InlayHint[] | null> {
        const request: InlayHintsParams = {
            textDocument: { uri: documentUri },
        };
        const client = Server.client;
        return client
            .onReady()
            .then(() =>
                client.sendRequest<InlayHint[] | null>(
                    'rust-analyzer/inlayHints',
                    request,
                ),
            );
    }
}
