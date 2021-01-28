//! rust-analyzer extensions to the LSP.

use std::{collections::HashMap, path::PathBuf};

use lsp_types::request::Request;
use lsp_types::{
    notification::Notification, CodeActionKind, Position, Range, TextDocumentIdentifier,
};
use serde::{Deserialize, Serialize};

pub enum AnalyzerStatus {}

impl Request for AnalyzerStatus {
    type Params = AnalyzerStatusParams;
    type Result = String;
    const METHOD: &'static str = "rust-analyzer/analyzerStatus";
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzerStatusParams {
    pub text_document: Option<TextDocumentIdentifier>,
}

pub enum MemoryUsage {}

impl Request for MemoryUsage {
    type Params = ();
    type Result = String;
    const METHOD: &'static str = "rust-analyzer/memoryUsage";
}

pub enum ReloadWorkspace {}

impl Request for ReloadWorkspace {
    type Params = ();
    type Result = ();
    const METHOD: &'static str = "rust-analyzer/reloadWorkspace";
}

pub enum SyntaxTree {}

impl Request for SyntaxTree {
    type Params = SyntaxTreeParams;
    type Result = String;
    const METHOD: &'static str = "rust-analyzer/syntaxTree";
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SyntaxTreeParams {
    pub text_document: TextDocumentIdentifier,
    pub range: Option<Range>,
}

pub enum ViewHir {}

impl Request for ViewHir {
    type Params = lsp_types::TextDocumentPositionParams;
    type Result = String;
    const METHOD: &'static str = "rust-analyzer/viewHir";
}

pub enum ExpandMacro {}

impl Request for ExpandMacro {
    type Params = ExpandMacroParams;
    type Result = Option<ExpandedMacro>;
    const METHOD: &'static str = "rust-analyzer/expandMacro";
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ExpandMacroParams {
    pub text_document: TextDocumentIdentifier,
    pub position: Position,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ExpandedMacro {
    pub name: String,
    pub expansion: String,
}

pub enum MatchingBrace {}

impl Request for MatchingBrace {
    type Params = MatchingBraceParams;
    type Result = Vec<Position>;
    const METHOD: &'static str = "experimental/matchingBrace";
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct MatchingBraceParams {
    pub text_document: TextDocumentIdentifier,
    pub positions: Vec<Position>,
}

pub enum ParentModule {}

impl Request for ParentModule {
    type Params = lsp_types::TextDocumentPositionParams;
    type Result = Option<lsp_types::GotoDefinitionResponse>;
    const METHOD: &'static str = "experimental/parentModule";
}

pub enum JoinLines {}

impl Request for JoinLines {
    type Params = JoinLinesParams;
    type Result = Vec<lsp_types::TextEdit>;
    const METHOD: &'static str = "experimental/joinLines";
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct JoinLinesParams {
    pub text_document: TextDocumentIdentifier,
    pub ranges: Vec<Range>,
}

pub enum OnEnter {}

impl Request for OnEnter {
    type Params = lsp_types::TextDocumentPositionParams;
    type Result = Option<Vec<SnippetTextEdit>>;
    const METHOD: &'static str = "experimental/onEnter";
}

pub enum Runnables {}

impl Request for Runnables {
    type Params = RunnablesParams;
    type Result = Vec<Runnable>;
    const METHOD: &'static str = "experimental/runnables";
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct RunnablesParams {
    pub text_document: TextDocumentIdentifier,
    pub position: Option<Position>,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Runnable {
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<lsp_types::LocationLink>,
    pub kind: RunnableKind,
    pub args: CargoRunnable,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum RunnableKind {
    Cargo,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct CargoRunnable {
    // command to be executed instead of cargo
    pub override_cargo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_root: Option<PathBuf>,
    // command, --package and --lib stuff
    pub cargo_args: Vec<String>,
    // user-specified additional cargo args, like `--release`.
    pub cargo_extra_args: Vec<String>,
    // stuff after --
    pub executable_args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expect_test: Option<bool>,
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
    ParameterHint,
    ChainingHint,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct InlayHint {
    pub range: Range,
    pub kind: InlayKind,
    pub label: String,
}

pub enum Ssr {}

impl Request for Ssr {
    type Params = SsrParams;
    type Result = lsp_types::WorkspaceEdit;
    const METHOD: &'static str = "experimental/ssr";
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SsrParams {
    pub query: String,
    pub parse_only: bool,

    /// File position where SSR was invoked. Paths in `query` will be resolved relative to this
    /// position.
    #[serde(flatten)]
    pub position: lsp_types::TextDocumentPositionParams,

    /// Current selections. Search/replace will be restricted to these if non-empty.
    pub selections: Vec<lsp_types::Range>,
}

pub enum StatusNotification {}

#[serde(rename_all = "camelCase")]
#[derive(Serialize, Deserialize)]
pub enum Status {
    Loading,
    ReadyPartial,
    Ready,
    NeedsReload,
    Invalid,
}

#[derive(Deserialize, Serialize)]
pub struct StatusParams {
    pub status: Status,
}

impl Notification for StatusNotification {
    type Params = StatusParams;
    const METHOD: &'static str = "rust-analyzer/status";
}

pub enum CodeActionRequest {}

impl Request for CodeActionRequest {
    type Params = lsp_types::CodeActionParams;
    type Result = Option<Vec<CodeAction>>;
    const METHOD: &'static str = "textDocument/codeAction";
}

pub enum CodeActionResolveRequest {}
impl Request for CodeActionResolveRequest {
    type Params = CodeAction;
    type Result = CodeAction;
    const METHOD: &'static str = "codeAction/resolve";
}

#[derive(Debug, PartialEq, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeAction {
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<CodeActionKind>,
    // We don't handle commands on the client-side
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub command: Option<lsp_types::Command>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edit: Option<SnippetWorkspaceEdit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_preferred: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<CodeActionData>,
}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeActionData {
    pub code_action_params: lsp_types::CodeActionParams,
    pub id: String,
}

#[derive(Debug, Eq, PartialEq, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SnippetWorkspaceEdit {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub changes: Option<HashMap<lsp_types::Url, Vec<lsp_types::TextEdit>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_changes: Option<Vec<SnippetDocumentChangeOperation>>,
}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(untagged, rename_all = "lowercase")]
pub enum SnippetDocumentChangeOperation {
    Op(lsp_types::ResourceOp),
    Edit(SnippetTextDocumentEdit),
}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SnippetTextDocumentEdit {
    pub text_document: lsp_types::OptionalVersionedTextDocumentIdentifier,
    pub edits: Vec<SnippetTextEdit>,
}

#[derive(Debug, Eq, PartialEq, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SnippetTextEdit {
    pub range: Range,
    pub new_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub insert_text_format: Option<lsp_types::InsertTextFormat>,
}

pub enum HoverRequest {}

impl Request for HoverRequest {
    type Params = lsp_types::HoverParams;
    type Result = Option<Hover>;
    const METHOD: &'static str = "textDocument/hover";
}

#[derive(Debug, PartialEq, Clone, Deserialize, Serialize)]
pub struct Hover {
    #[serde(flatten)]
    pub hover: lsp_types::Hover,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<CommandLinkGroup>,
}

#[derive(Debug, PartialEq, Clone, Default, Deserialize, Serialize)]
pub struct CommandLinkGroup {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub commands: Vec<CommandLink>,
}

// LSP v3.15 Command does not have a `tooltip` field, vscode supports one.
#[derive(Debug, PartialEq, Clone, Default, Deserialize, Serialize)]
pub struct CommandLink {
    #[serde(flatten)]
    pub command: lsp_types::Command,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tooltip: Option<String>,
}

pub enum ExternalDocs {}

impl Request for ExternalDocs {
    type Params = lsp_types::TextDocumentPositionParams;
    type Result = Option<lsp_types::Url>;
    const METHOD: &'static str = "experimental/externalDocs";
}

pub enum OpenCargoToml {}

impl Request for OpenCargoToml {
    type Params = OpenCargoTomlParams;
    type Result = Option<lsp_types::GotoDefinitionResponse>;
    const METHOD: &'static str = "experimental/openCargoToml";
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct OpenCargoTomlParams {
    pub text_document: TextDocumentIdentifier,
}
