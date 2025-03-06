import * as anser from "anser";
import * as vscode from "vscode";
import {
    type ProviderResult,
    Range,
    type TextEditorDecorationType,
    ThemeColor,
    window,
} from "vscode";
import type { Ctx } from "./ctx";
import { unwrapUndefinable } from "./util";

export const URI_SCHEME = "rust-analyzer-diagnostics-view";

export class TextDocumentProvider implements vscode.TextDocumentContentProvider {
    private _onDidChange = new vscode.EventEmitter<vscode.Uri>();

    public constructor(private readonly ctx: Ctx) {}

    get onDidChange(): vscode.Event<vscode.Uri> {
        return this._onDidChange.event;
    }

    triggerUpdate(uri: vscode.Uri) {
        if (uri.scheme === URI_SCHEME) {
            this._onDidChange.fire(uri);
        }
    }

    dispose() {
        this._onDidChange.dispose();
    }

    async provideTextDocumentContent(uri: vscode.Uri): Promise<string> {
        const contents = getRenderedDiagnostic(this.ctx, uri);
        return anser.ansiToText(contents);
    }
}

function getRenderedDiagnostic(ctx: Ctx, uri: vscode.Uri): string {
    const diags = ctx.client?.diagnostics?.get(vscode.Uri.parse(uri.fragment, true));
    if (!diags) {
        return "Unable to find original rustc diagnostic";
    }

    const diag = diags[parseInt(uri.query)];
    if (!diag) {
        return "Unable to find original rustc diagnostic";
    }
    const rendered = (diag as unknown as { data?: { rendered?: string } }).data?.rendered;

    if (!rendered) {
        return "Unable to find original rustc diagnostic";
    }

    return rendered;
}

interface AnserStyle {
    fg: string;
    bg: string;
    fg_truecolor: string;
    bg_truecolor: string;
    decorations: Array<anser.DecorationName>;
}

export class AnsiDecorationProvider implements vscode.Disposable {
    private _decorationTypes = new Map<AnserStyle, TextEditorDecorationType>();

    public constructor(private readonly ctx: Ctx) {}

    dispose(): void {
        for (const decorationType of this._decorationTypes.values()) {
            decorationType.dispose();
        }

        this._decorationTypes.clear();
    }

    async provideDecorations(editor: vscode.TextEditor) {
        if (editor.document.uri.scheme !== URI_SCHEME) {
            return;
        }

        const decorations = (await this._getDecorations(editor.document.uri)) || [];
        for (const [decorationType, ranges] of decorations) {
            editor.setDecorations(decorationType, ranges);
        }
    }

    private _getDecorations(
        uri: vscode.Uri,
    ): ProviderResult<[TextEditorDecorationType, Range[]][]> {
        const stringContents = getRenderedDiagnostic(this.ctx, uri);
        const lines = stringContents.split("\n");

        const result = new Map<TextEditorDecorationType, Range[]>();
        // Populate all known decoration types in the result. This forces any
        // lingering decorations to be cleared if the text content changes to
        // something without ANSI codes for a given decoration type.
        for (const decorationType of this._decorationTypes.values()) {
            result.set(decorationType, []);
        }

        for (const [lineNumber, line] of lines.entries()) {
            const totalEscapeLength = 0;
            const parsed = anser.ansiToJson(line, { use_classes: true });
            let offset = 0;

            for (const span of parsed) {
                const { content, ...style } = span;

                const range = new Range(
                    lineNumber,
                    offset - totalEscapeLength,
                    lineNumber,
                    offset + content.length - totalEscapeLength,
                );

                offset += content.length;

                const decorationType = this._getDecorationType(style);

                if (!result.has(decorationType)) {
                    result.set(decorationType, []);
                }

                result.get(decorationType)!.push(range);
            }
        }

        return [...result];
    }

    private _getDecorationType(style: AnserStyle): TextEditorDecorationType {
        let decorationType = this._decorationTypes.get(style);

        if (decorationType) {
            return decorationType;
        }

        const fontWeight = style.decorations.find((s) => s === "bold");
        const fontStyle = style.decorations.find((s) => s === "italic");
        const textDecoration = style.decorations.find((s) => s === "underline");

        decorationType = window.createTextEditorDecorationType({
            backgroundColor: AnsiDecorationProvider._convertColor(style.bg, style.bg_truecolor),
            color: AnsiDecorationProvider._convertColor(style.fg, style.fg_truecolor),
            fontWeight,
            fontStyle,
            textDecoration,
        });

        this._decorationTypes.set(style, decorationType);

        return decorationType;
    }

    // NOTE: This could just be a kebab-case to camelCase conversion, but I think it's
    // a short enough list to just write these by hand
    static readonly _anserToThemeColor: Record<string, ThemeColor> = {
        "ansi-black": new ThemeColor("terminal.ansiBlack"),
        "ansi-white": new ThemeColor("terminal.ansiWhite"),
        "ansi-red": new ThemeColor("terminal.ansiRed"),
        "ansi-green": new ThemeColor("terminal.ansiGreen"),
        "ansi-yellow": new ThemeColor("terminal.ansiYellow"),
        "ansi-blue": new ThemeColor("terminal.ansiBlue"),
        "ansi-magenta": new ThemeColor("terminal.ansiMagenta"),
        "ansi-cyan": new ThemeColor("terminal.ansiCyan"),

        "ansi-bright-black": new ThemeColor("terminal.ansiBrightBlack"),
        "ansi-bright-white": new ThemeColor("terminal.ansiBrightWhite"),
        "ansi-bright-red": new ThemeColor("terminal.ansiBrightRed"),
        "ansi-bright-green": new ThemeColor("terminal.ansiBrightGreen"),
        "ansi-bright-yellow": new ThemeColor("terminal.ansiBrightYellow"),
        "ansi-bright-blue": new ThemeColor("terminal.ansiBrightBlue"),
        "ansi-bright-magenta": new ThemeColor("terminal.ansiBrightMagenta"),
        "ansi-bright-cyan": new ThemeColor("terminal.ansiBrightCyan"),
    };

    private static _convertColor(
        color?: string,
        truecolor?: string,
    ): ThemeColor | string | undefined {
        if (!color) {
            return undefined;
        }

        if (color === "ansi-truecolor") {
            if (!truecolor) {
                return undefined;
            }
            return `rgb(${truecolor})`;
        }

        const paletteMatch = color.match(/ansi-palette-(.+)/);
        if (paletteMatch) {
            const paletteColor = paletteMatch[1];
            // anser won't return both the RGB and the color name at the same time,
            // so just fake a single foreground control char with the palette number:
            const spans = anser.ansiToJson(`\x1b[38;5;${paletteColor}m`);
            const span = unwrapUndefinable(spans[1]);
            const rgb = span.fg;

            if (rgb) {
                return `rgb(${rgb})`;
            }
        }

        return AnsiDecorationProvider._anserToThemeColor[color];
    }
}
