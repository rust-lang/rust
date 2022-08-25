//! base_db defines basic database traits. The concrete DB is defined by ide.

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

mod input;
mod change;
pub mod fixture;

use std::{panic, sync::Arc};

use stdx::hash::NoHashHashSet;
use syntax::{ast, Parse, SourceFile, TextRange, TextSize};

pub use crate::{
    change::Change,
    input::{
        CrateData, CrateDisplayName, CrateGraph, CrateId, CrateName, CrateOrigin, Dependency,
        Edition, Env, LangCrateOrigin, ProcMacro, ProcMacroExpander, ProcMacroExpansionError,
        ProcMacroId, ProcMacroKind, ProcMacroLoadResult, SourceRoot, SourceRootId,
    },
};
pub use salsa::{self, Cancelled};
pub use vfs::{file_set::FileSet, AnchoredPath, AnchoredPathBuf, FileId, VfsPath};

#[macro_export]
macro_rules! impl_intern_key {
    ($name:ident) => {
        impl $crate::salsa::InternKey for $name {
            fn from_intern_id(v: $crate::salsa::InternId) -> Self {
                $name(v)
            }
            fn as_intern_id(&self) -> $crate::salsa::InternId {
                self.0
            }
        }
    };
}

pub trait Upcast<T: ?Sized> {
    fn upcast(&self) -> &T;
}

#[derive(Clone, Copy, Debug)]
pub struct FilePosition {
    pub file_id: FileId,
    pub offset: TextSize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FileRange {
    pub file_id: FileId,
    pub range: TextRange,
}

pub const DEFAULT_LRU_CAP: usize = 128;

pub trait FileLoader {
    /// Text of the file.
    fn file_text(&self, file_id: FileId) -> Arc<String>;
    fn resolve_path(&self, path: AnchoredPath<'_>) -> Option<FileId>;
    fn relevant_crates(&self, file_id: FileId) -> Arc<NoHashHashSet<CrateId>>;
}

/// Database which stores all significant input facts: source code and project
/// model. Everything else in rust-analyzer is derived from these queries.
#[salsa::query_group(SourceDatabaseStorage)]
pub trait SourceDatabase: FileLoader + std::fmt::Debug {
    // Parses the file into the syntax tree.
    #[salsa::invoke(parse_query)]
    fn parse(&self, file_id: FileId) -> Parse<ast::SourceFile>;

    /// The crate graph.
    #[salsa::input]
    fn crate_graph(&self) -> Arc<CrateGraph>;
}

fn parse_query(db: &dyn SourceDatabase, file_id: FileId) -> Parse<ast::SourceFile> {
    let _p = profile::span("parse_query").detail(|| format!("{:?}", file_id));
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
    /// Source root of the file.
    #[salsa::input]
    fn file_source_root(&self, file_id: FileId) -> SourceRootId;
    /// Contents of the source root.
    #[salsa::input]
    fn source_root(&self, id: SourceRootId) -> Arc<SourceRoot>;

    fn source_root_crates(&self, id: SourceRootId) -> Arc<NoHashHashSet<CrateId>>;
}

fn source_root_crates(db: &dyn SourceDatabaseExt, id: SourceRootId) -> Arc<NoHashHashSet<CrateId>> {
    let graph = db.crate_graph();
    let res = graph
        .iter()
        .filter(|&krate| {
            let root_file = graph[krate].root_file_id;
            db.file_source_root(root_file) == id
        })
        .collect();
    Arc::new(res)
}

/// Silly workaround for cyclic deps between the traits
pub struct FileLoaderDelegate<T>(pub T);

impl<T: SourceDatabaseExt> FileLoader for FileLoaderDelegate<&'_ T> {
    fn file_text(&self, file_id: FileId) -> Arc<String> {
        SourceDatabaseExt::file_text(self.0, file_id)
    }
    fn resolve_path(&self, path: AnchoredPath<'_>) -> Option<FileId> {
        // FIXME: this *somehow* should be platform agnostic...
        let source_root = self.0.file_source_root(path.anchor);
        let source_root = self.0.source_root(source_root);
        source_root.resolve_path(path)
    }

    fn relevant_crates(&self, file_id: FileId) -> Arc<NoHashHashSet<CrateId>> {
        let _p = profile::span("relevant_crates");
        let source_root = self.0.file_source_root(file_id);
        self.0.source_root_crates(source_root)
    }
}
