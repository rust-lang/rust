import * as vscode from "vscode";

import { isRustEditor, setContextValue } from "./util";
import type { CtxInit } from "./ctx";
import * as ra from "./lsp_ext";

export class SyntaxTreeProvider implements vscode.TreeDataProvider<SyntaxElement> {
    private _onDidChangeTreeData: vscode.EventEmitter<SyntaxElement | undefined | void> =
        new vscode.EventEmitter<SyntaxElement | undefined | void>();
    readonly onDidChangeTreeData: vscode.Event<SyntaxElement | undefined | void> =
        this._onDidChangeTreeData.event;
    ctx: CtxInit;
    root: SyntaxNode | undefined;
    hideWhitespace: boolean = false;

    constructor(ctx: CtxInit) {
        this.ctx = ctx;
    }

    getTreeItem(element: SyntaxElement): vscode.TreeItem {
        return new SyntaxTreeItem(element);
    }

    getChildren(element?: SyntaxElement): vscode.ProviderResult<SyntaxElement[]> {
        return this.getRawChildren(element);
    }

    getParent(element: SyntaxElement): vscode.ProviderResult<SyntaxElement> {
        return element.parent;
    }

    resolveTreeItem(
        item: SyntaxTreeItem,
        element: SyntaxElement,
        _token: vscode.CancellationToken,
    ): vscode.ProviderResult<SyntaxTreeItem> {
        const editor = vscode.window.activeTextEditor;

        if (editor !== undefined) {
            const start = editor.document.positionAt(element.start);
            const end = editor.document.positionAt(element.end);
            const range = new vscode.Range(start, end);

            const text = editor.document.getText(range);
            item.tooltip = new vscode.MarkdownString().appendCodeblock(text, "rust");
        }

        return item;
    }

    private getRawChildren(element?: SyntaxElement): SyntaxElement[] {
        if (element?.type === "Node") {
            if (this.hideWhitespace) {
                return element.children.filter((e) => e.kind !== "WHITESPACE");
            }

            return element.children;
        }

        if (element?.type === "Token") {
            return [];
        }

        if (element === undefined && this.root !== undefined) {
            return [this.root];
        }

        return [];
    }

    async refresh(): Promise<void> {
        const editor = vscode.window.activeTextEditor;

        if (editor && isRustEditor(editor)) {
            const params = { textDocument: { uri: editor.document.uri.toString() }, range: null };
            const fileText = await this.ctx.client.sendRequest(ra.viewSyntaxTree, params);
            this.root = JSON.parse(fileText, (_key, value: SyntaxElement) => {
                if (value.type === "Node") {
                    for (const child of value.children) {
                        child.parent = value;
                    }
                }

                return value;
            });
        } else {
            this.root = undefined;
        }

        this._onDidChangeTreeData.fire();
    }

    getElementByRange(start: number, end: number): SyntaxElement | undefined {
        if (this.root === undefined) {
            return undefined;
        }

        let result: SyntaxElement = this.root;

        if (this.root.start === start && this.root.end === end) {
            return result;
        }

        let children = this.getRawChildren(this.root);

        outer: while (true) {
            for (const child of children) {
                if (child.start <= start && child.end >= end) {
                    result = child;
                    if (start === end && start === child.end) {
                        // When the cursor is on the very end of a token,
                        // we assume the user wants the next token instead.
                        continue;
                    }

                    if (child.type === "Token") {
                        return result;
                    } else {
                        children = this.getRawChildren(child);
                        continue outer;
                    }
                }
            }

            return result;
        }
    }

    async toggleWhitespace() {
        this.hideWhitespace = !this.hideWhitespace;
        this._onDidChangeTreeData.fire();
        await setContextValue("rustSyntaxTree.hideWhitespace", this.hideWhitespace);
    }
}

export type SyntaxNode = {
    type: "Node";
    kind: string;
    start: number;
    end: number;
    istart?: number;
    iend?: number;
    children: SyntaxElement[];
    parent?: SyntaxElement;
};

type SyntaxToken = {
    type: "Token";
    kind: string;
    start: number;
    end: number;
    istart?: number;
    iend?: number;
    parent?: SyntaxElement;
};

export type SyntaxElement = SyntaxNode | SyntaxToken;

export class SyntaxTreeItem extends vscode.TreeItem {
    constructor(private readonly element: SyntaxElement) {
        super(element.kind);
        const icon = getIcon(element.kind);
        if (element.type === "Node") {
            this.contextValue = "syntaxNode";
            this.iconPath = icon ?? new vscode.ThemeIcon("list-tree");
            this.collapsibleState = vscode.TreeItemCollapsibleState.Expanded;
        } else {
            this.contextValue = "syntaxToken";
            this.iconPath = icon ?? new vscode.ThemeIcon("symbol-string");
            this.collapsibleState = vscode.TreeItemCollapsibleState.None;
        }

        if (element.istart !== undefined && element.iend !== undefined) {
            this.description = `${this.element.istart}..${this.element.iend}`;
        } else {
            this.description = `${this.element.start}..${this.element.end}`;
        }
    }
}

function getIcon(kind: string): vscode.ThemeIcon | undefined {
    const icon = iconTable[kind];

    if (icon !== undefined) {
        return icon;
    }

    if (kind.endsWith("_KW")) {
        return new vscode.ThemeIcon(
            "symbol-keyword",
            new vscode.ThemeColor("symbolIcon.keywordForeground"),
        );
    }

    if (operators.includes(kind)) {
        return new vscode.ThemeIcon(
            "symbol-operator",
            new vscode.ThemeColor("symbolIcon.operatorForeground"),
        );
    }

    return undefined;
}

const iconTable: Record<string, vscode.ThemeIcon> = {
    CALL_EXPR: new vscode.ThemeIcon("call-outgoing"),
    COMMENT: new vscode.ThemeIcon("comment"),
    ENUM: new vscode.ThemeIcon("symbol-enum", new vscode.ThemeColor("symbolIcon.enumForeground")),
    FN: new vscode.ThemeIcon(
        "symbol-function",
        new vscode.ThemeColor("symbolIcon.functionForeground"),
    ),
    FLOAT_NUMBER: new vscode.ThemeIcon(
        "symbol-number",
        new vscode.ThemeColor("symbolIcon.numberForeground"),
    ),
    INDEX_EXPR: new vscode.ThemeIcon(
        "symbol-array",
        new vscode.ThemeColor("symbolIcon.arrayForeground"),
    ),
    INT_NUMBER: new vscode.ThemeIcon(
        "symbol-number",
        new vscode.ThemeColor("symbolIcon.numberForeground"),
    ),
    LITERAL: new vscode.ThemeIcon(
        "symbol-misc",
        new vscode.ThemeColor("symbolIcon.miscForeground"),
    ),
    MODULE: new vscode.ThemeIcon(
        "symbol-module",
        new vscode.ThemeColor("symbolIcon.moduleForeground"),
    ),
    METHOD_CALL_EXPR: new vscode.ThemeIcon("call-outgoing"),
    PARAM: new vscode.ThemeIcon(
        "symbol-parameter",
        new vscode.ThemeColor("symbolIcon.parameterForeground"),
    ),
    RECORD_FIELD: new vscode.ThemeIcon(
        "symbol-field",
        new vscode.ThemeColor("symbolIcon.fieldForeground"),
    ),
    SOURCE_FILE: new vscode.ThemeIcon("file-code"),
    STRING: new vscode.ThemeIcon("quote"),
    STRUCT: new vscode.ThemeIcon(
        "symbol-struct",
        new vscode.ThemeColor("symbolIcon.structForeground"),
    ),
    TRAIT: new vscode.ThemeIcon(
        "symbol-interface",
        new vscode.ThemeColor("symbolIcon.interfaceForeground"),
    ),
    TYPE_PARAM: new vscode.ThemeIcon(
        "symbol-type-parameter",
        new vscode.ThemeColor("symbolIcon.typeParameterForeground"),
    ),
    VARIANT: new vscode.ThemeIcon(
        "symbol-enum-member",
        new vscode.ThemeColor("symbolIcon.enumMemberForeground"),
    ),
    WHITESPACE: new vscode.ThemeIcon("whitespace"),
};

const operators = [
    "PLUS",
    "PLUSEQ",
    "MINUS",
    "MINUSEQ",
    "STAR",
    "STAREQ",
    "SLASH",
    "SLASHEQ",
    "PERCENT",
    "PERCENTEQ",
    "CARET",
    "CARETEQ",
    "AMP",
    "AMPEQ",
    "AMP2",
    "PIPE",
    "PIPEEQ",
    "PIPE2",
    "SHL",
    "SHLEQ",
    "SHR",
    "SHREQ",
    "EQ",
    "EQ2",
    "BANG",
    "NEQ",
    "L_ANGLE",
    "LTEQ",
    "R_ANGLE",
    "GTEQ",
    "COLON2",
    "THIN_ARROW",
    "FAT_ARROW",
    "DOT",
    "DOT2",
    "DOT2EQ",
    "AT",
];
