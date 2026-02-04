//! Bidirectional protocol messages

use std::{
    io::{self, BufRead, Write},
    ops::Range,
};

use paths::Utf8PathBuf;
use serde::{Deserialize, Serialize};

use crate::{
    ProcMacroKind,
    legacy_protocol::msg::{FlatTree, Message, PanicMessage, ServerConfig},
    transport::postcard,
};

#[derive(Debug, Serialize, Deserialize)]
pub enum SubRequest {
    FilePath { file_id: u32 },
    SourceText { file_id: u32, ast_id: u32, start: u32, end: u32 },
    LocalFilePath { file_id: u32 },
    LineColumn { file_id: u32, ast_id: u32, offset: u32 },
    ByteRange { file_id: u32, ast_id: u32, start: u32, end: u32 },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum SubResponse {
    FilePathResult {
        name: String,
    },
    SourceTextResult {
        text: Option<String>,
    },
    LocalFilePathResult {
        name: Option<String>,
    },
    /// Line and column are 1-based.
    LineColumnResult {
        line: u32,
        column: u32,
    },
    ByteRangeResult {
        range: Range<usize>,
    },
    Cancel {
        reason: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum BidirectionalMessage {
    Request(Request),
    Response(Response),
    SubRequest(SubRequest),
    SubResponse(SubResponse),
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    ListMacros { dylib_path: Utf8PathBuf },
    ExpandMacro(Box<ExpandMacro>),
    ApiVersionCheck {},
    SetConfig(ServerConfig),
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    ListMacros(Result<Vec<(String, ProcMacroKind)>, String>),
    ExpandMacro(Result<FlatTree, PanicMessage>),
    ApiVersionCheck(u32),
    SetConfig(ServerConfig),
    ExpandMacroExtended(Result<ExpandMacroExtended, PanicMessage>),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExpandMacro {
    pub lib: Utf8PathBuf,
    pub env: Vec<(String, String)>,
    pub current_dir: Option<String>,
    pub data: ExpandMacroData,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExpandMacroExtended {
    pub tree: FlatTree,
    pub span_data_table: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExpandMacroData {
    pub macro_body: FlatTree,
    pub macro_name: String,
    pub attributes: Option<FlatTree>,
    #[serde(default)]
    pub has_global_spans: ExpnGlobals,
    #[serde(default)]
    pub span_data_table: Vec<u32>,
}

#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize)]
pub struct ExpnGlobals {
    pub def_site: usize,
    pub call_site: usize,
    pub mixed_site: usize,
}

impl Message for BidirectionalMessage {
    type Buf = Vec<u8>;

    fn read(inp: &mut dyn BufRead, buf: &mut Self::Buf) -> io::Result<Option<Self>> {
        Ok(match postcard::read(inp, buf)? {
            None => None,
            Some(buf) => Some(postcard::decode(buf)?),
        })
    }
    fn write(self, out: &mut dyn Write) -> io::Result<()> {
        let value = postcard::encode(&self)?;
        postcard::write(out, &value)
    }
}
