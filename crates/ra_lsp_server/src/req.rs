//! Defines `rust-analyzer` specific custom messages.

use lsp_types::{Location, Position, Range, TextDocumentIdentifier, Url};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

pub use lsp_types::{
    notification::*, request::*, ApplyWorkspaceEditParams, CodeActionParams, CodeLens,
    CodeLensParams, CompletionParams, CompletionResponse, DidChangeConfigurationParams,
    DidChangeWatchedFilesParams, DidChangeWatchedFilesRegistrationOptions,
    DocumentOnTypeFormattingParams, DocumentSymbolParams, DocumentSymbolResponse,
    FileSystemWatcher, Hover, InitializeResult, MessageType, PublishDiagnosticsParams,
    ReferenceParams, Registration, RegistrationParams, SelectionRange, SelectionRangeParams,
    ShowMessageParams, SignatureHelp, TextDocumentEdit, TextDocumentPositionParams, TextEdit,
    WorkspaceEdit, WorkspaceSymbolParams,
};

pub enum AnalyzerStatus {}

impl Request for AnalyzerStatus {
    type Params = ();
    type Result = String;
    const METHOD: &'static str = "rust-analyzer/analyzerStatus";
}

pub enum CollectGarbage {}

impl Request for CollectGarbage {
    type Params = ();
    type Result = ();
    const METHOD: &'static str = "rust-analyzer/collectGarbage";
}

pub enum SyntaxTree {}

impl Request for SyntaxTree {
    type Params = SyntaxTreeParams;
    type Result = String;
    const METHOD: &'static str = "rust-analyzer/syntaxTree";
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SyntaxTreeParams {
    pub text_document: TextDocumentIdentifier,
    pub range: Option<Range>,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ExpandedMacro {
    pub name: String,
    pub expansion: String,
}

pub enum ExpandMacro {}

impl Request for ExpandMacro {
    type Params = ExpandMacroParams;
    type Result = Option<ExpandedMacro>;
    const METHOD: &'static str = "rust-analyzer/expandMacro";
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ExpandMacroParams {
    pub text_document: TextDocumentIdentifier,
    pub position: Option<Position>,
}

pub enum FindMatchingBrace {}

impl Request for FindMatchingBrace {
    type Params = FindMatchingBraceParams;
    type Result = Vec<Position>;
    const METHOD: &'static str = "rust-analyzer/findMatchingBrace";
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct FindMatchingBraceParams {
    pub text_document: TextDocumentIdentifier,
    pub offsets: Vec<Position>,
}

pub enum DecorationsRequest {}

impl Request for DecorationsRequest {
    type Params = TextDocumentIdentifier;
    type Result = Vec<Decoration>;
    const METHOD: &'static str = "rust-analyzer/decorationsRequest";
}

pub enum PublishDecorations {}

impl Notification for PublishDecorations {
    type Params = PublishDecorationsParams;
    const METHOD: &'static str = "rust-analyzer/publishDecorations";
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct PublishDecorationsParams {
    pub uri: Url,
    pub decorations: Vec<Decoration>,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Decoration {
    pub range: Range,
    pub tag: &'static str,
    pub binding_hash: Option<String>,
}

pub enum ParentModule {}

impl Request for ParentModule {
    type Params = TextDocumentPositionParams;
    type Result = Vec<Location>;
    const METHOD: &'static str = "rust-analyzer/parentModule";
}

pub enum JoinLines {}

impl Request for JoinLines {
    type Params = JoinLinesParams;
    type Result = SourceChange;
    const METHOD: &'static str = "rust-analyzer/joinLines";
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct JoinLinesParams {
    pub text_document: TextDocumentIdentifier,
    pub range: Range,
}

pub enum OnEnter {}

impl Request for OnEnter {
    type Params = TextDocumentPositionParams;
    type Result = Option<SourceChange>;
    const METHOD: &'static str = "rust-analyzer/onEnter";
}

pub enum Runnables {}

impl Request for Runnables {
    type Params = RunnablesParams;
    type Result = Vec<Runnable>;
    const METHOD: &'static str = "rust-analyzer/runnables";
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct RunnablesParams {
    pub text_document: TextDocumentIdentifier,
    pub position: Option<Position>,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Runnable {
    pub range: Range,
    pub label: String,
    pub bin: String,
    pub args: Vec<String>,
    pub env: FxHashMap<String, String>,
    pub cwd: Option<String>,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SourceChange {
    pub label: String,
    pub workspace_edit: WorkspaceEdit,
    pub cursor_position: Option<TextDocumentPositionParams>,
}

pub enum InlayHints {}

impl Request for InlayHints {
    type Params = InlayHintsParams;
    type Result = Vec<InlayHint>;
    const METHOD: &'static str = "rust-analyzer/inlayHints";
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct InlayHintsParams {
    pub text_document: TextDocumentIdentifier,
}

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum InlayKind {
    TypeHint,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct InlayHint {
    pub range: Range,
    pub kind: InlayKind,
    pub label: String,
}
