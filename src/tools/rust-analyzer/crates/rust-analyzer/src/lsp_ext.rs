//! rust-analyzer extensions to the LSP.

use std::{collections::HashMap, path::PathBuf};

use lsp_types::request::Request;
use lsp_types::{
    notification::Notification, CodeActionKind, DocumentOnTypeFormattingParams,
    PartialResultParams, Position, Range, TextDocumentIdentifier, WorkDoneProgressParams,
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

pub enum ShuffleCrateGraph {}

impl Request for ShuffleCrateGraph {
    type Params = ();
    type Result = ();
    const METHOD: &'static str = "rust-analyzer/shuffleCrateGraph";
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

pub enum ViewFileText {}

impl Request for ViewFileText {
    type Params = lsp_types::TextDocumentIdentifier;
    type Result = String;
    const METHOD: &'static str = "rust-analyzer/viewFileText";
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ViewCrateGraphParams {
    /// Include *all* crates, not just crates in the workspace.
    pub full: bool,
}

pub enum ViewCrateGraph {}

impl Request for ViewCrateGraph {
    type Params = ViewCrateGraphParams;
    type Result = String;
    const METHOD: &'static str = "rust-analyzer/viewCrateGraph";
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ViewItemTreeParams {
    pub text_document: TextDocumentIdentifier,
}

pub enum ViewItemTree {}

impl Request for ViewItemTree {
    type Params = ViewItemTreeParams;
    type Result = String;
    const METHOD: &'static str = "rust-analyzer/viewItemTree";
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

pub enum CancelFlycheck {}

impl Request for CancelFlycheck {
    type Params = ();
    type Result = ();
    const METHOD: &'static str = "rust-analyzer/cancelFlycheck";
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

pub enum RelatedTests {}

impl Request for RelatedTests {
    type Params = lsp_types::TextDocumentPositionParams;
    type Result = Vec<TestInfo>;
    const METHOD: &'static str = "rust-analyzer/relatedTests";
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TestInfo {
    pub runnable: Runnable,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct InlayHintsParams {
    pub text_document: TextDocumentIdentifier,
    pub range: Option<lsp_types::Range>,
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

pub enum ServerStatusNotification {}

impl Notification for ServerStatusNotification {
    type Params = ServerStatusParams;
    const METHOD: &'static str = "experimental/serverStatus";
}

#[derive(Deserialize, Serialize, PartialEq, Eq, Clone)]
pub struct ServerStatusParams {
    pub health: Health,
    pub quiescent: bool,
    pub message: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Health {
    Ok,
    Warning,
    Error,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<lsp_types::Command>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub change_annotations:
        Option<HashMap<lsp_types::ChangeAnnotationIdentifier, lsp_types::ChangeAnnotation>>,
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
    /// The annotation id if this is an annotated
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotation_id: Option<lsp_types::ChangeAnnotationIdentifier>,
}

pub enum HoverRequest {}

impl Request for HoverRequest {
    type Params = HoverParams;
    type Result = Option<Hover>;
    const METHOD: &'static str = "textDocument/hover";
}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HoverParams {
    pub text_document: TextDocumentIdentifier,
    pub position: PositionOrRange,

    #[serde(flatten)]
    pub work_done_progress_params: WorkDoneProgressParams,
}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum PositionOrRange {
    Position(lsp_types::Position),
    Range(lsp_types::Range),
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

/// Information about CodeLens, that is to be resolved.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum CodeLensResolveData {
    Impls(lsp_types::request::GotoImplementationParams),
    References(lsp_types::TextDocumentPositionParams),
}

pub fn supports_utf8(caps: &lsp_types::ClientCapabilities) -> bool {
    caps.offset_encoding.as_deref().unwrap_or_default().iter().any(|it| it == "utf-8")
}

pub enum MoveItem {}

impl Request for MoveItem {
    type Params = MoveItemParams;
    type Result = Vec<SnippetTextEdit>;
    const METHOD: &'static str = "experimental/moveItem";
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct MoveItemParams {
    pub direction: MoveItemDirection,
    pub text_document: TextDocumentIdentifier,
    pub range: Range,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum MoveItemDirection {
    Up,
    Down,
}

#[derive(Debug)]
pub enum WorkspaceSymbol {}

impl Request for WorkspaceSymbol {
    type Params = WorkspaceSymbolParams;
    type Result = Option<Vec<lsp_types::SymbolInformation>>;
    const METHOD: &'static str = "workspace/symbol";
}

#[derive(Debug, Eq, PartialEq, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceSymbolParams {
    #[serde(flatten)]
    pub partial_result_params: PartialResultParams,

    #[serde(flatten)]
    pub work_done_progress_params: WorkDoneProgressParams,

    /// A non-empty query string
    pub query: String,

    pub search_scope: Option<WorkspaceSymbolSearchScope>,

    pub search_kind: Option<WorkspaceSymbolSearchKind>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum WorkspaceSymbolSearchScope {
    Workspace,
    WorkspaceAndDependencies,
}

#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum WorkspaceSymbolSearchKind {
    OnlyTypes,
    AllSymbols,
}

/// The document on type formatting request is sent from the client to
/// the server to format parts of the document during typing.  This is
/// almost same as lsp_types::request::OnTypeFormatting, but the
/// result has SnippetTextEdit in it instead of TextEdit.
#[derive(Debug)]
pub enum OnTypeFormatting {}

impl Request for OnTypeFormatting {
    type Params = DocumentOnTypeFormattingParams;
    type Result = Option<Vec<SnippetTextEdit>>;
    const METHOD: &'static str = "textDocument/onTypeFormatting";
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionResolveData {
    pub position: lsp_types::TextDocumentPositionParams,
    pub imports: Vec<CompletionImport>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InlayHintResolveData {
    pub text_document: TextDocumentIdentifier,
    pub position: PositionOrRange,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionImport {
    pub full_import_path: String,
    pub imported_name: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct ClientCommandOptions {
    pub commands: Vec<String>,
}
