//! ra_db defines basic database traits. The concrete DB is defined by ra_ide_api.
mod cancellation;
mod syntax_ptr;
mod input;
mod loc2id;
pub mod mock;

use std::panic;

use ra_syntax::{TextUnit, TextRange, SourceFile, TreeArc};

pub use crate::{
    cancellation::{Canceled, Cancelable},
    syntax_ptr::LocalSyntaxPtr,
    input::{
        FilesDatabase, FileId, CrateId, SourceRoot, SourceRootId, CrateGraph, Dependency,
        FileTextQuery, FileSourceRootQuery, SourceRootQuery, LocalRootsQuery, LibraryRootsQuery, CrateGraphQuery,
        FileRelativePathQuery
    },
    loc2id::LocationIntener,
};

pub trait BaseDatabase: salsa::Database + panic::RefUnwindSafe {
    fn check_canceled(&self) -> Cancelable<()> {
        self.salsa_runtime()
            .if_current_revision_is_canceled(Canceled::throw);
        Ok(())
    }

    fn catch_canceled<F: FnOnce(&Self) -> T + panic::UnwindSafe, T>(
        &self,
        f: F,
    ) -> Result<T, Canceled> {
        panic::catch_unwind(|| f(self)).map_err(|err| match err.downcast::<Canceled>() {
            Ok(canceled) => *canceled,
            Err(payload) => panic::resume_unwind(payload),
        })
    }
}

salsa::query_group! {
    pub trait SyntaxDatabase: crate::input::FilesDatabase + BaseDatabase {
        fn source_file(file_id: FileId) -> TreeArc<SourceFile> {
            type SourceFileQuery;
        }
    }
}

fn source_file(db: &impl SyntaxDatabase, file_id: FileId) -> TreeArc<SourceFile> {
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
