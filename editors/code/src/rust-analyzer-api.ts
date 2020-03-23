/**
 * This file mirrors `crates/rust-analyzer/src/req.rs` declarations.
 */

import * as lc from "vscode-languageclient";

type Option<T> = null | T;
type Vec<T> = T[];
type FxHashMap<K extends PropertyKey, V> = Record<K, V>;

function request<TParams, TResult>(method: string) {
    return new lc.RequestType<TParams, TResult, unknown>(`rust-analyzer/${method}`);
}
function notification<TParam>(method: string) {
    return new lc.NotificationType<TParam>(method);
}


export const analyzerStatus = request<null, string>("analyzerStatus");


export const collectGarbage = request<null, null>("collectGarbage");


export interface SyntaxTreeParams {
    textDocument: lc.TextDocumentIdentifier;
    range: Option<lc.Range>;
}
export const syntaxTree = request<SyntaxTreeParams, string>("syntaxTree");


export interface ExpandMacroParams {
    textDocument: lc.TextDocumentIdentifier;
    position: Option<lc.Position>;
}
export interface ExpandedMacro {
    name: string;
    expansion: string;
}
export const expandMacro = request<ExpandMacroParams, Option<ExpandedMacro>>("expandMacro");


export interface FindMatchingBraceParams {
    textDocument: lc.TextDocumentIdentifier;
    offsets: Vec<lc.Position>;
}
export const findMatchingBrace = request<FindMatchingBraceParams, Vec<lc.Position>>("findMatchingBrace");


export interface PublishDecorationsParams {
    uri: string;
    decorations: Vec<Decoration>;
}
export interface Decoration {
    range: lc.Range;
    tag: string;
    bindingHash: Option<string>;
}
export const decorationsRequest = request<lc.TextDocumentIdentifier, Vec<Decoration>>("decorationsRequest");


export const parentModule = request<lc.TextDocumentPositionParams, Vec<lc.Location>>("parentModule");


export interface JoinLinesParams {
    textDocument: lc.TextDocumentIdentifier;
    range: lc.Range;
}
export const joinLines = request<JoinLinesParams, SourceChange>("joinLines");


export const onEnter = request<lc.TextDocumentPositionParams, Option<SourceChange>>("onEnter");

export interface RunnablesParams {
    textDocument: lc.TextDocumentIdentifier;
    position: Option<lc.Position>;
}
export interface Runnable {
    range: lc.Range;
    label: string;
    bin: string;
    args: Vec<string>;
    extraArgs: Vec<string>;
    env: FxHashMap<string, string>;
    cwd: Option<string>;
}
export const runnables = request<RunnablesParams, Vec<Runnable>>("runnables");

export type InlayHint = InlayHint.TypeHint | InlayHint.ParamHint | InlayHint.ChainingHint;

export namespace InlayHint {
    export const enum Kind {
        TypeHint = "TypeHint",
        ParamHint = "ParameterHint",
        ChainingHint = "ChainingHint",
    }
    interface Common {
        range: lc.Range;
        label: string;
    }
    export type TypeHint = Common & { kind: Kind.TypeHint };
    export type ParamHint = Common & { kind: Kind.ParamHint };
    export type ChainingHint = Common & { kind: Kind.ChainingHint };
}
export interface InlayHintsParams {
    textDocument: lc.TextDocumentIdentifier;
}
export const inlayHints = request<InlayHintsParams, Vec<InlayHint>>("inlayHints");


export interface SsrParams {
    query: string;
    parseOnly: boolean;
}
export const ssr = request<SsrParams, SourceChange>("ssr");


export const publishDecorations = notification<PublishDecorationsParams>("publishDecorations");


export interface SourceChange {
    label: string;
    workspaceEdit: lc.WorkspaceEdit;
    cursorPosition: Option<lc.TextDocumentPositionParams>;
}
