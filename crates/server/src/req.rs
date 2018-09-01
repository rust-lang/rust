use std::collections::HashMap;
use languageserver_types::{TextDocumentIdentifier, Range, Url, Position, Location};
use url_serde;

pub use languageserver_types::{
    request::*, notification::*,
    InitializeResult, PublishDiagnosticsParams,
    DocumentSymbolParams, DocumentSymbolResponse,
    CodeActionParams, ApplyWorkspaceEditParams,
    ExecuteCommandParams,
    WorkspaceSymbolParams,
    TextDocumentPositionParams,
    TextEdit,
    CompletionParams, CompletionResponse,
    DocumentOnTypeFormattingParams,
    TextDocumentEdit,
};

pub enum SyntaxTree {}

impl Request for SyntaxTree {
    type Params = SyntaxTreeParams;
    type Result = String;
    const METHOD: &'static str = "m/syntaxTree";
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SyntaxTreeParams {
    pub text_document: TextDocumentIdentifier
}

pub enum ExtendSelection {}

impl Request for ExtendSelection {
    type Params = ExtendSelectionParams;
    type Result = ExtendSelectionResult;
    const METHOD: &'static str = "m/extendSelection";
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ExtendSelectionParams {
    pub text_document: TextDocumentIdentifier,
    pub selections: Vec<Range>,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ExtendSelectionResult {
    pub selections: Vec<Range>,
}

pub enum FindMatchingBrace {}

impl Request for FindMatchingBrace {
    type Params = FindMatchingBraceParams;
    type Result = Vec<Position>;
    const METHOD: &'static str = "m/findMatchingBrace";
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
    const METHOD: &'static str = "m/decorationsRequest";
}

pub enum PublishDecorations {}

impl Notification for PublishDecorations {
    type Params = PublishDecorationsParams;
    const METHOD: &'static str = "m/publishDecorations";
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct PublishDecorationsParams {
    #[serde(with = "url_serde")]
    pub uri: Url,
    pub decorations: Vec<Decoration>,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Decoration {
    pub range: Range,
    pub tag: &'static str
}

pub enum ParentModule {}

impl Request for ParentModule {
    type Params = TextDocumentIdentifier;
    type Result = Vec<Location>;
    const METHOD: &'static str = "m/parentModule";
}

pub enum JoinLines {}

impl Request for JoinLines {
    type Params = JoinLinesParams;
    type Result = SourceChange;
    const METHOD: &'static str = "m/joinLines";
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct JoinLinesParams {
    pub text_document: TextDocumentIdentifier,
    pub range: Range,
}

pub enum Runnables {}

impl Request for Runnables {
    type Params = RunnablesParams;
    type Result = Vec<Runnable>;
    const METHOD: &'static str = "m/runnables";
}

#[derive(Deserialize, Debug)]
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
    pub env: HashMap<String, String>,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SourceChange {
    pub label: String,
    pub source_file_edits: Vec<TextDocumentEdit>,
    pub file_system_edits: Vec<FileSystemEdit>,
    pub cursor_position: Option<TextDocumentPositionParams>,
}

#[derive(Serialize, Debug)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum FileSystemEdit {
    CreateFile {
        #[serde(with = "url_serde")]
        uri: Url
    },
    MoveFile {
        #[serde(with = "url_serde")]
        src: Url,
        #[serde(with = "url_serde")]
        dst: Url,
    }
}
