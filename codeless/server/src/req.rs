use languageserver_types::TextDocumentIdentifier;
pub use languageserver_types::request::*;
pub use languageserver_types::{InitializeResult};

pub enum SyntaxTree {}

impl Request for SyntaxTree {
    type Params = SyntaxTreeParams;
    type Result = String;
    const METHOD: &'static str = "m/syntaxTree";
}

#[derive(Deserialize, Debug)]
#[serde(rename_all="camelCase")]
pub struct SyntaxTreeParams {
    pub text_document: TextDocumentIdentifier
}
