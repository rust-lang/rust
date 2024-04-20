import * as vscode from "vscode";

import type { Ctx, Disposable } from "./ctx";
import { type RustEditor, isRustEditor } from "./util";
import { unwrapUndefinable } from "./undefinable";

// FIXME: consider implementing this via the Tree View API?
// https://code.visualstudio.com/api/extension-guides/tree-view
export class AstInspector implements vscode.HoverProvider, vscode.DefinitionProvider, Disposable {
    private readonly astDecorationType = vscode.window.createTextEditorDecorationType({
        borderColor: new vscode.ThemeColor("rust_analyzer.syntaxTreeBorder"),
        borderStyle: "solid",
        borderWidth: "2px",
    });
    private rustEditor: undefined | RustEditor;

    // Lazy rust token range -> syntax tree file range.
    private readonly rust2Ast = new Lazy(() => {
        const astEditor = this.findAstTextEditor();
        if (!this.rustEditor || !astEditor) return undefined;

        const buf: [vscode.Range, vscode.Range][] = [];
        for (let i = 0; i < astEditor.document.lineCount; ++i) {
            const astLine = astEditor.document.lineAt(i);

            // Heuristically look for nodes with quoted text (which are token nodes)
            const isTokenNode = astLine.text.lastIndexOf('"') >= 0;
            if (!isTokenNode) continue;

            const rustRange = this.parseRustTextRange(this.rustEditor.document, astLine.text);
            if (!rustRange) continue;

            buf.push([rustRange, this.findAstNodeRange(astLine)]);
        }
        return buf;
    });

    constructor(ctx: Ctx) {
        ctx.pushExtCleanup(
            vscode.languages.registerHoverProvider({ scheme: "rust-analyzer" }, this),
        );
        ctx.pushExtCleanup(vscode.languages.registerDefinitionProvider({ language: "rust" }, this));
        vscode.workspace.onDidCloseTextDocument(
            this.onDidCloseTextDocument,
            this,
            ctx.subscriptions,
        );
        vscode.workspace.onDidChangeTextDocument(
            this.onDidChangeTextDocument,
            this,
            ctx.subscriptions,
        );
        vscode.window.onDidChangeVisibleTextEditors(
            this.onDidChangeVisibleTextEditors,
            this,
            ctx.subscriptions,
        );
    }
    dispose() {
        this.setRustEditor(undefined);
    }

    private onDidChangeTextDocument(event: vscode.TextDocumentChangeEvent) {
        if (
            this.rustEditor &&
            event.document.uri.toString() === this.rustEditor.document.uri.toString()
        ) {
            this.rust2Ast.reset();
        }
    }

    private onDidCloseTextDocument(doc: vscode.TextDocument) {
        if (this.rustEditor && doc.uri.toString() === this.rustEditor.document.uri.toString()) {
            this.setRustEditor(undefined);
        }
    }

    private onDidChangeVisibleTextEditors(editors: readonly vscode.TextEditor[]) {
        if (!this.findAstTextEditor()) {
            this.setRustEditor(undefined);
            return;
        }
        this.setRustEditor(editors.find(isRustEditor));
    }

    private findAstTextEditor(): undefined | vscode.TextEditor {
        return vscode.window.visibleTextEditors.find(
            (it) => it.document.uri.scheme === "rust-analyzer",
        );
    }

    private setRustEditor(newRustEditor: undefined | RustEditor) {
        if (this.rustEditor && this.rustEditor !== newRustEditor) {
            this.rustEditor.setDecorations(this.astDecorationType, []);
            this.rust2Ast.reset();
        }
        this.rustEditor = newRustEditor;
    }

    // additional positional params are omitted
    provideDefinition(
        doc: vscode.TextDocument,
        pos: vscode.Position,
    ): vscode.ProviderResult<vscode.DefinitionLink[]> {
        if (!this.rustEditor || doc.uri.toString() !== this.rustEditor.document.uri.toString()) {
            return;
        }

        const astEditor = this.findAstTextEditor();
        if (!astEditor) return;

        const rust2AstRanges = this.rust2Ast
            .get()
            ?.find(([rustRange, _]) => rustRange.contains(pos));
        if (!rust2AstRanges) return;

        const [rustFileRange, astFileRange] = rust2AstRanges;

        astEditor.revealRange(astFileRange);
        astEditor.selection = new vscode.Selection(astFileRange.start, astFileRange.end);

        return [
            {
                targetRange: astFileRange,
                targetUri: astEditor.document.uri,
                originSelectionRange: rustFileRange,
                targetSelectionRange: astFileRange,
            },
        ];
    }

    // additional positional params are omitted
    provideHover(
        doc: vscode.TextDocument,
        hoverPosition: vscode.Position,
    ): vscode.ProviderResult<vscode.Hover> {
        if (!this.rustEditor) return;

        const astFileLine = doc.lineAt(hoverPosition.line);

        const rustFileRange = this.parseRustTextRange(this.rustEditor.document, astFileLine.text);
        if (!rustFileRange) return;

        this.rustEditor.setDecorations(this.astDecorationType, [rustFileRange]);
        this.rustEditor.revealRange(rustFileRange);

        const rustSourceCode = this.rustEditor.document.getText(rustFileRange);
        const astFileRange = this.findAstNodeRange(astFileLine);

        return new vscode.Hover(["```rust\n" + rustSourceCode + "\n```"], astFileRange);
    }

    private findAstNodeRange(astLine: vscode.TextLine): vscode.Range {
        const lineOffset = astLine.range.start;
        const begin = lineOffset.translate(undefined, astLine.firstNonWhitespaceCharacterIndex);
        const end = lineOffset.translate(undefined, astLine.text.trimEnd().length);
        return new vscode.Range(begin, end);
    }

    private parseRustTextRange(
        doc: vscode.TextDocument,
        astLine: string,
    ): undefined | vscode.Range {
        const parsedRange = /(\d+)\.\.(\d+)/.exec(astLine);
        if (!parsedRange) return;

        const [begin, end] = parsedRange.slice(1).map((off) => this.positionAt(doc, +off));
        const actualBegin = unwrapUndefinable(begin);
        const actualEnd = unwrapUndefinable(end);
        return new vscode.Range(actualBegin, actualEnd);
    }

    // Memoize the last value, otherwise the CPU is at 100% single core
    // with quadratic lookups when we build rust2Ast cache
    cache?: { doc: vscode.TextDocument; offset: number; line: number };

    positionAt(doc: vscode.TextDocument, targetOffset: number): vscode.Position {
        if (doc.eol === vscode.EndOfLine.LF) {
            return doc.positionAt(targetOffset);
        }

        // Dirty workaround for crlf line endings
        // We are still in this prehistoric era of carriage returns here...

        let line = 0;
        let offset = 0;

        const cache = this.cache;
        if (cache?.doc === doc && cache.offset <= targetOffset) {
            ({ line, offset } = cache);
        }

        while (true) {
            const lineLenWithLf = doc.lineAt(line).text.length + 1;
            if (offset + lineLenWithLf > targetOffset) {
                this.cache = { doc, offset, line };
                return doc.positionAt(targetOffset + line);
            }
            offset += lineLenWithLf;
            line += 1;
        }
    }
}

class Lazy<T> {
    val: undefined | T;

    constructor(private readonly compute: () => undefined | T) {}

    get() {
        return this.val ?? (this.val = this.compute());
    }

    reset() {
        this.val = undefined;
    }
}
