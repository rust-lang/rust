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
import { unwrapUndefinable } from "./undefinable";

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

            // eslint-disable-next-line camelcase
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
        "ansi-black": "ansiBlack",
        "ansi-white": "ansiWhite",
        "ansi-red": "ansiRed",
        "ansi-green": "ansiGreen",
        "ansi-yellow": "ansiYellow",
        "ansi-blue": "ansiBlue",
        "ansi-magenta": "ansiMagenta",
        "ansi-cyan": "ansiCyan",

        "ansi-bright-black": "ansiBrightBlack",
        "ansi-bright-white": "ansiBrightWhite",
        "ansi-bright-red": "ansiBrightRed",
        "ansi-bright-green": "ansiBrightGreen",
        "ansi-bright-yellow": "ansiBrightYellow",
        "ansi-bright-blue": "ansiBrightBlue",
        "ansi-bright-magenta": "ansiBrightMagenta",
        "ansi-bright-cyan": "ansiBrightCyan",
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

        const themeColor = AnsiDecorationProvider._anserToThemeColor[color];
        if (themeColor) {
            return new ThemeColor("terminal." + themeColor);
        }

        return undefined;
    }
}
