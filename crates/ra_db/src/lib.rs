//! ra_db defines basic database traits. The concrete DB is defined by ra_ide_api.
mod cancellation;
mod input;
mod loc2id;
pub mod mock;

use std::panic;

use ra_syntax::{TextUnit, TextRange, SourceFile, TreeArc};

pub use ::salsa as salsa;
pub use crate::{
    cancellation::Canceled,
    input::{
        FilesDatabase, FileId, CrateId, SourceRoot, SourceRootId, CrateGraph, Dependency,
        FileTextQuery, FileSourceRootQuery, SourceRootQuery, SourceRootCratesQuery, LocalRootsQuery, LibraryRootsQuery, CrateGraphQuery,
        FileRelativePathQuery
    },
    loc2id::LocationIntener,
};

pub trait BaseDatabase: salsa::Database + panic::RefUnwindSafe {
    /// Aborts current query if there are pending changes.
    ///
    /// rust-analyzer needs to be able to answer semantic questions about the
    /// code while the code is being modified. A common problem is that a
    /// long-running query is being calculated when a new change arrives.
    ///
    /// We can't just apply the change immediately: this will cause the pending
    /// query to see inconsistent state (it will observe an absence of
    /// repeatable read). So what we do is we **cancel** all pending queries
    /// before applying the change.
    ///
    /// We implement cancellation by panicking with a special value and catching
    /// it on the API boundary. Salsa explicitly supports this use-case.
    fn check_canceled(&self) {
        if self.salsa_runtime().is_current_revision_canceled() {
            Canceled::throw()
        }
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

#[salsa::query_group]
pub trait SyntaxDatabase: crate::input::FilesDatabase + BaseDatabase {
    fn source_file(&self, file_id: FileId) -> TreeArc<SourceFile>;
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
