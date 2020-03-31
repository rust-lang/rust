import * as vscode from 'vscode';
import * as ra from '../rust-analyzer-api';

import { Ctx, Cmd, Disposable } from '../ctx';
import { isRustDocument, RustEditor, isRustEditor, sleep } from '../util';

const AST_FILE_SCHEME = "rust-analyzer";

// Opens the virtual file that will show the syntax tree
//
// The contents of the file come from the `TextDocumentContentProvider`
export function syntaxTree(ctx: Ctx): Cmd {
    const tdcp = new TextDocumentContentProvider(ctx);

    void new AstInspector(ctx);

    ctx.pushCleanup(vscode.workspace.registerTextDocumentContentProvider(AST_FILE_SCHEME, tdcp));

    return async () => {
        const editor = vscode.window.activeTextEditor;
        const rangeEnabled = !!editor && !editor.selection.isEmpty;

        const uri = rangeEnabled
            ? vscode.Uri.parse(`${tdcp.uri.toString()}?range=true`)
            : tdcp.uri;

        const document = await vscode.workspace.openTextDocument(uri);

        tdcp.eventEmitter.fire(uri);

        void await vscode.window.showTextDocument(document, {
            viewColumn: vscode.ViewColumn.Two,
            preserveFocus: true
        });
    };
}

class TextDocumentContentProvider implements vscode.TextDocumentContentProvider {
    readonly uri = vscode.Uri.parse('rust-analyzer://syntaxtree');
    readonly eventEmitter = new vscode.EventEmitter<vscode.Uri>();


    constructor(private readonly ctx: Ctx) {
        vscode.workspace.onDidChangeTextDocument(this.onDidChangeTextDocument, this, ctx.subscriptions);
        vscode.window.onDidChangeActiveTextEditor(this.onDidChangeActiveTextEditor, this, ctx.subscriptions);
    }

    private onDidChangeTextDocument(event: vscode.TextDocumentChangeEvent) {
        if (isRustDocument(event.document)) {
            // We need to order this after language server updates, but there's no API for that.
            // Hence, good old sleep().
            void sleep(10).then(() => this.eventEmitter.fire(this.uri));
        }
    }
    private onDidChangeActiveTextEditor(editor: vscode.TextEditor | undefined) {
        if (editor && isRustEditor(editor)) {
            this.eventEmitter.fire(this.uri);
        }
    }

    provideTextDocumentContent(uri: vscode.Uri, ct: vscode.CancellationToken): vscode.ProviderResult<string> {
        const rustEditor = this.ctx.activeRustEditor;
        if (!rustEditor) return '';

        // When the range based query is enabled we take the range of the selection
        const range = uri.query === 'range=true' && !rustEditor.selection.isEmpty
            ? this.ctx.client.code2ProtocolConverter.asRange(rustEditor.selection)
            : null;

        const params = { textDocument: { uri: rustEditor.document.uri.toString() }, range, };
        return this.ctx.client.sendRequest(ra.syntaxTree, params, ct);
    }

    get onDidChange(): vscode.Event<vscode.Uri> {
        return this.eventEmitter.event;
    }
}


// FIXME: consider implementing this via the Tree View API?
// https://code.visualstudio.com/api/extension-guides/tree-view
class AstInspector implements vscode.HoverProvider, Disposable {
    private static readonly astDecorationType = vscode.window.createTextEditorDecorationType({
        fontStyle: "normal",
        border: "#ffffff 1px solid",
    });
    private rustEditor: undefined | RustEditor;

    constructor(ctx: Ctx) {
        ctx.pushCleanup(vscode.languages.registerHoverProvider({ scheme: AST_FILE_SCHEME }, this));
        vscode.workspace.onDidCloseTextDocument(this.onDidCloseTextDocument, this, ctx.subscriptions);
        vscode.window.onDidChangeVisibleTextEditors(this.onDidChangeVisibleTextEditors, this, ctx.subscriptions);

        ctx.pushCleanup(this);
    }
    dispose() {
        this.setRustEditor(undefined);
    }

    private onDidCloseTextDocument(doc: vscode.TextDocument) {
        if (!!this.rustEditor && doc.uri.toString() === this.rustEditor.document.uri.toString()) {
            this.setRustEditor(undefined);
        }
    }

    private onDidChangeVisibleTextEditors(editors: vscode.TextEditor[]) {
        if (editors.every(suspect => suspect.document.uri.scheme !== AST_FILE_SCHEME)) {
            this.setRustEditor(undefined);
            return;
        }
        this.setRustEditor(editors.find(isRustEditor));
    }

    private setRustEditor(newRustEditor: undefined | RustEditor) {
        if (newRustEditor !== this.rustEditor) {
            this.rustEditor?.setDecorations(AstInspector.astDecorationType, []);
        }
        this.rustEditor = newRustEditor;
    }

    provideHover(doc: vscode.TextDocument, hoverPosition: vscode.Position): vscode.ProviderResult<vscode.Hover> {
        if (!this.rustEditor) return;

        const astTextLine = doc.lineAt(hoverPosition.line);

        const rustTextRange = this.parseRustTextRange(this.rustEditor.document, astTextLine.text);
        if (!rustTextRange) return;

        this.rustEditor.setDecorations(AstInspector.astDecorationType, [rustTextRange]);
        this.rustEditor.revealRange(rustTextRange);

        const rustSourceCode = this.rustEditor.document.getText(rustTextRange);
        const astTextRange = this.findAstRange(astTextLine);

        return new vscode.Hover(["```rust\n" + rustSourceCode + "\n```"], astTextRange);
    }

    private findAstRange(astLine: vscode.TextLine) {
        const lineOffset = astLine.range.start;
        const begin = lineOffset.translate(undefined, astLine.firstNonWhitespaceCharacterIndex);
        const end = lineOffset.translate(undefined, astLine.text.trimEnd().length);
        return new vscode.Range(begin, end);
    }

    private parseRustTextRange(doc: vscode.TextDocument, astLine: string): undefined | vscode.Range {
        const parsedRange = /\[(\d+); (\d+)\)/.exec(astLine);
        if (!parsedRange) return;

        const [begin, end] = parsedRange.slice(1).map(off => doc.positionAt(+off));

        return new vscode.Range(begin, end);
    }
}
