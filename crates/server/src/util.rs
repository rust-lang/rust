use std::path::PathBuf;
use languageserver_types::{TextDocumentItem, VersionedTextDocumentIdentifier,
                           TextDocumentIdentifier, Url};
use ::{Result};

pub trait FilePath {
    fn file_path(&self) -> Result<PathBuf>;
}

impl FilePath for TextDocumentItem {
    fn file_path(&self) -> Result<PathBuf> {
        self.uri.file_path()
    }
}

impl FilePath for VersionedTextDocumentIdentifier {
    fn file_path(&self) -> Result<PathBuf> {
        self.uri.file_path()
    }
}

impl FilePath for TextDocumentIdentifier {
    fn file_path(&self) -> Result<PathBuf> {
        self.uri.file_path()
    }
}

impl FilePath for Url {
    fn file_path(&self) -> Result<PathBuf> {
        self.to_file_path()
            .map_err(|()| format_err!("invalid uri: {}", self))
    }
}
