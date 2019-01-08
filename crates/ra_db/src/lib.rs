//! ra_db defines basic database traits. Concrete DB is defined by ra_ide_api.
mod cancelation;
mod syntax_ptr;
mod input;
mod loc2id;
pub mod mock;

use ra_syntax::{TextUnit, TextRange, SourceFile, TreePtr};

pub use crate::{
    cancelation::{Canceled, Cancelable},
    syntax_ptr::LocalSyntaxPtr,
    input::{
        FilesDatabase, FileId, CrateId, SourceRoot, SourceRootId, CrateGraph, Dependency,
        FileTextQuery, FileSourceRootQuery, SourceRootQuery, LocalRootsQuery, LibraryRootsQuery, CrateGraphQuery,
        FileRelativePathQuery
    },
    loc2id::LocationIntener,
};

pub trait BaseDatabase: salsa::Database {
    fn check_canceled(&self) -> Cancelable<()> {
        if self.salsa_runtime().is_current_revision_canceled() {
            Err(Canceled::new())
        } else {
            Ok(())
        }
    }
}

salsa::query_group! {
    pub trait SyntaxDatabase: crate::input::FilesDatabase + BaseDatabase {
        fn source_file(file_id: FileId) -> TreePtr<SourceFile> {
            type SourceFileQuery;
        }
    }
}

fn source_file(db: &impl SyntaxDatabase, file_id: FileId) -> TreePtr<SourceFile> {
    let text = db.file_text(file_id);
    SourceFile::parse(&*text)
}

#[derive(Clone, Copy, Debug)]
pub struct FilePosition {
    pub file_id: FileId,
    pub offset: TextUnit,
}

#[derive(Clone, Copy, Debug)]
pub struct FileRange {
    pub file_id: FileId,
    pub range: TextRange,
}
