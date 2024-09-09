//! Custom LSP definitions and protocol conversions.

use core::fmt;

pub mod ext;

pub(crate) mod capabilities;
pub(crate) mod from_proto;
pub(crate) mod semantic_tokens;
pub(crate) mod to_proto;
pub(crate) mod utils;

#[derive(Debug)]
pub(crate) struct LspError {
    pub(crate) code: i32,
    pub(crate) message: String,
}

impl LspError {
    pub(crate) fn new(code: i32, message: String) -> LspError {
        LspError { code, message }
    }
}

impl fmt::Display for LspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Language Server request failed with {}. ({})", self.code, self.message)
    }
}

impl std::error::Error for LspError {}
