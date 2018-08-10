use serde::{ser::Serialize, de::DeserializeOwned};
use languageserver_types::{TextDocumentIdentifier, Range};

pub use languageserver_types::{
    request::*, notification::*,
    InitializeResult, PublishDiagnosticsParams,
};


pub trait ClientRequest: 'static {
    type Params: DeserializeOwned + Send + 'static;
    type Result: Serialize + Send + 'static;
    const METHOD: &'static str;
}

impl<T> ClientRequest for T
    where T: Request + 'static,
          T::Params: DeserializeOwned + Send + 'static,
          T::Result: Serialize + Send + 'static,
{
    type Params = <T as Request>::Params;
    type Result = <T as Request>::Result;
    const METHOD: &'static str = <T as Request>::METHOD;
}


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
