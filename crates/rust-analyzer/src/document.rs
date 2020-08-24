//! In-memory document information.

/// Information about a document that the Language Client
/// knows about.
/// Its lifetime is driven by the textDocument/didOpen and textDocument/didClose
/// client notifications.
#[derive(Debug, Clone)]
pub(crate) struct DocumentData {
    pub version: Option<i64>,
}

impl DocumentData {
    pub fn new(version: i64) -> Self {
        DocumentData { version: Some(version) }
    }
}
