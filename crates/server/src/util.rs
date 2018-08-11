use std::path::PathBuf;
use languageserver_types::{TextDocumentItem, VersionedTextDocumentIdentifier,
                           TextDocumentIdentifier, Url};
use ::{Result};

pub trait FnBox<A, R>: Send {
    fn call_box(self: Box<Self>, a: A) -> R;
}

impl<A, R, F: FnOnce(A) -> R + Send> FnBox<A, R> for F {
    fn call_box(self: Box<F>, a: A) -> R {
        (*self)(a)
    }
}

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
