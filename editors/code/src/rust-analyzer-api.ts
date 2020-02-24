/**
 * This file mirrors `crates/rust-analyzer/src/req.rs` declarations.
 */

import { RequestType, TextDocumentIdentifier, Position, Range, TextDocumentPositionParams, Location, NotificationType, WorkspaceEdit } from "vscode-languageclient";

type Option<T> = null | T;
type Vec<T> = T[];
type FxHashMap<K extends PropertyKey, V> = Record<K, V>;

function request<TParams, TResult>(method: string) {
    return new RequestType<TParams, TResult, unknown>(`rust-analyzer/${method}`);
}
function notification<TParam>(method: string) {
    return new NotificationType<TParam>(method);
}


export const analyzerStatus = request<null, string>("analyzerStatus");


export const collectGarbage = request<null, null>("collectGarbage");


export interface SyntaxTreeParams {
    textDocument: TextDocumentIdentifier;
    range: Option<Range>;
}
export const syntaxTree = request<SyntaxTreeParams, string>("syntaxTree");


export interface ExpandMacroParams {
    textDocument: TextDocumentIdentifier;
    position: Option<Position>;
}
export interface ExpandedMacro {
    name: string;
    expansion: string;
}
export const expandMacro = request<ExpandMacroParams, Option<ExpandedMacro>>("expandMacro");


export interface FindMatchingBraceParams {
    textDocument: TextDocumentIdentifier;
    offsets: Vec<Position>;
}
export const findMatchingBrace = request<FindMatchingBraceParams, Vec<Position>>("findMatchingBrace");


export interface PublishDecorationsParams {
    uri: string;
    decorations: Vec<Decoration>;
}
export interface Decoration {
    range: Range;
    tag: string;
    bindingHash: Option<string>;
}
export const decorationsRequest = request<TextDocumentIdentifier, Vec<Decoration>>("decorationsRequest");


export const parentModule = request<TextDocumentPositionParams, Vec<Location>>("parentModule");


export interface JoinLinesParams {
    textDocument: TextDocumentIdentifier;
    range: Range;
}
export const joinLines = request<JoinLinesParams, SourceChange>("joinLines");


export const onEnter = request<TextDocumentPositionParams, Option<SourceChange>>("onEnter");

export interface RunnablesParams {
    textDocument: TextDocumentIdentifier;
    position: Option<Position>;
}
export interface Runnable {
    range: Range;
    label: string;
    bin: string;
    args: Vec<string>;
    env: FxHashMap<string, string>;
    cwd: Option<string>;
}
export const runnables = request<RunnablesParams, Vec<Runnable>>("runnables");


export const enum InlayKind {
    TypeHint = "TypeHint",
    ParameterHint = "ParameterHint",
}
export interface InlayHint {
    range: Range;
    kind: InlayKind;
    label: string;
}
export interface InlayHintsParams {
    textDocument: TextDocumentIdentifier;
}
export const inlayHints = request<InlayHintsParams, Vec<InlayHint>>("inlayHints");


export interface SsrParams {
    arg: string;
}
export const ssr = request<SsrParams, SourceChange>("ssr");


export const publishDecorations = notification<PublishDecorationsParams>("publishDecorations");


export interface SourceChange {
    label: string;
    workspaceEdit: WorkspaceEdit;
    cursorPosition: Option<TextDocumentPositionParams>;
}
