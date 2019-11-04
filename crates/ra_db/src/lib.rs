//! ra_db defines basic database traits. The concrete DB is defined by ra_ide_api.
mod cancellation;
mod input;
pub mod fixture;

use std::{panic, sync::Arc};

use ra_prof::profile;
use ra_syntax::{ast, Parse, SourceFile, TextRange, TextUnit};

pub use crate::{
    cancellation::Canceled,
    input::{CrateGraph, CrateId, Dependency, Edition, FileId, SourceRoot, SourceRootId},
};
pub use relative_path::{RelativePath, RelativePathBuf};
pub use salsa;

pub trait CheckCanceled {
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
    fn check_canceled(&self);

    fn catch_canceled<F, T>(&self, f: F) -> Result<T, Canceled>
    where
        Self: Sized + panic::RefUnwindSafe,
        F: FnOnce(&Self) -> T + panic::UnwindSafe,
    {
        panic::catch_unwind(|| f(self)).map_err(|err| match err.downcast::<Canceled>() {
            Ok(canceled) => *canceled,
            Err(payload) => panic::resume_unwind(payload),
        })
    }
}

impl<T: salsa::Database> CheckCanceled for T {
    fn check_canceled(&self) {
        if self.salsa_runtime().is_current_revision_canceled() {
            Canceled::throw()
        }
    }
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

pub const DEFAULT_LRU_CAP: usize = 128;

pub trait FileLoader {
    /// Text of the file.
    fn file_text(&self, file_id: FileId) -> Arc<String>;
    fn resolve_relative_path(&self, anchor: FileId, relative_path: &RelativePath)
        -> Option<FileId>;
    fn relevant_crates(&self, file_id: FileId) -> Arc<Vec<CrateId>>;
}

/// Database which stores all significant input facts: source code and project
/// model. Everything else in rust-analyzer is derived from these queries.
#[salsa::query_group(SourceDatabaseStorage)]
pub trait SourceDatabase: CheckCanceled + FileLoader + std::fmt::Debug {
    // Parses the file into the syntax tree.
    #[salsa::invoke(parse_query)]
    fn parse(&self, file_id: FileId) -> Parse<ast::SourceFile>;

    /// The crate graph.
    #[salsa::input]
    fn crate_graph(&self) -> Arc<CrateGraph>;
}

fn parse_query(db: &impl SourceDatabase, file_id: FileId) -> Parse<ast::SourceFile> {
    let _p = profile("parse_query");
    let text = db.file_text(file_id);
    SourceFile::parse(&*text)
}

/// We don't want to give HIR knowledge of source roots, hence we extract these
/// methods into a separate DB.
#[salsa::query_group(SourceDatabaseExtStorage)]
pub trait SourceDatabaseExt: SourceDatabase {
    #[salsa::input]
    fn file_text(&self, file_id: FileId) -> Arc<String>;
    /// Path to a file, relative to the root of its source root.
    #[salsa::input]
    fn file_relative_path(&self, file_id: FileId) -> RelativePathBuf;
    /// Source root of the file.
    #[salsa::input]
    fn file_source_root(&self, file_id: FileId) -> SourceRootId;
    /// Contents of the source root.
    #[salsa::input]
    fn source_root(&self, id: SourceRootId) -> Arc<SourceRoot>;

    fn source_root_crates(&self, id: SourceRootId) -> Arc<Vec<CrateId>>;
}

fn source_root_crates(
    db: &(impl SourceDatabaseExt + SourceDatabase),
    id: SourceRootId,
) -> Arc<Vec<CrateId>> {
    let root = db.source_root(id);
    let graph = db.crate_graph();
    let res = root.walk().filter_map(|it| graph.crate_id_for_crate_root(it)).collect::<Vec<_>>();
    Arc::new(res)
}

/// Silly workaround for cyclic deps between the traits
pub struct FileLoaderDelegate<T>(pub T);

impl<T: SourceDatabaseExt> FileLoader for FileLoaderDelegate<&'_ T> {
    fn file_text(&self, file_id: FileId) -> Arc<String> {
        SourceDatabaseExt::file_text(self.0, file_id)
    }
    fn resolve_relative_path(
        &self,
        anchor: FileId,
        relative_path: &RelativePath,
    ) -> Option<FileId> {
        let path = {
            let mut path = self.0.file_relative_path(anchor);
            assert!(path.pop());
            path.push(relative_path);
            path.normalize()
        };
        let source_root = self.0.file_source_root(anchor);
        let source_root = self.0.source_root(source_root);
        source_root.file_by_relative_path(&path)
    }

    fn relevant_crates(&self, file_id: FileId) -> Arc<Vec<CrateId>> {
        let source_root = self.0.file_source_root(file_id);
        self.0.source_root_crates(source_root)
    }
}
