//! base_db defines basic database traits. The concrete DB is defined by ide.

#![warn(rust_2018_idioms, unused_lifetimes)]

mod change;
mod input;

use std::panic;

use syntax::{ast, Parse, SourceFile};
use triomphe::Arc;

pub use crate::{
    change::FileChange,
    input::{
        CrateData, CrateDisplayName, CrateGraph, CrateId, CrateName, CrateOrigin, Dependency,
        DependencyKind, Edition, Env, LangCrateOrigin, ProcMacroPaths, ReleaseChannel, SourceRoot,
        SourceRootId, TargetLayoutLoadResult,
    },
};
pub use salsa::{self, Cancelled};
pub use span::{FilePosition, FileRange};
pub use vfs::{file_set::FileSet, AnchoredPath, AnchoredPathBuf, FileId, VfsPath};

pub use semver::{BuildMetadata, Prerelease, Version, VersionReq};

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

pub const DEFAULT_PARSE_LRU_CAP: usize = 128;
pub const DEFAULT_BORROWCK_LRU_CAP: usize = 1024;

pub trait FileLoader {
    /// Text of the file.
    fn file_text(&self, file_id: FileId) -> Arc<str>;
    fn resolve_path(&self, path: AnchoredPath<'_>) -> Option<FileId>;
    fn relevant_crates(&self, file_id: FileId) -> Arc<[CrateId]>;
}

/// Database which stores all significant input facts: source code and project
/// model. Everything else in rust-analyzer is derived from these queries.
#[salsa::query_group(SourceDatabaseStorage)]
pub trait SourceDatabase: FileLoader + std::fmt::Debug {
    /// Parses the file into the syntax tree.
    fn parse(&self, file_id: FileId) -> Parse<ast::SourceFile>;

    /// The crate graph.
    #[salsa::input]
    fn crate_graph(&self) -> Arc<CrateGraph>;

    // FIXME: Consider removing this, making HirDatabase::target_data_layout an input query
    #[salsa::input]
    fn data_layout(&self, krate: CrateId) -> TargetLayoutLoadResult;

    #[salsa::input]
    fn toolchain(&self, krate: CrateId) -> Option<Version>;

    #[salsa::transparent]
    fn toolchain_channel(&self, krate: CrateId) -> Option<ReleaseChannel>;
}

fn toolchain_channel(db: &dyn SourceDatabase, krate: CrateId) -> Option<ReleaseChannel> {
    db.toolchain(krate).as_ref().and_then(|v| ReleaseChannel::from_str(&v.pre))
}

fn parse(db: &dyn SourceDatabase, file_id: FileId) -> Parse<ast::SourceFile> {
    let _p = tracing::span!(tracing::Level::INFO, "parse_query", ?file_id).entered();
    let text = db.file_text(file_id);
    SourceFile::parse(&text)
}

/// We don't want to give HIR knowledge of source roots, hence we extract these
/// methods into a separate DB.
#[salsa::query_group(SourceDatabaseExtStorage)]
pub trait SourceDatabaseExt: SourceDatabase {
    #[salsa::input]
    fn file_text(&self, file_id: FileId) -> Arc<str>;
    /// Path to a file, relative to the root of its source root.
    /// Source root of the file.
    #[salsa::input]
    fn file_source_root(&self, file_id: FileId) -> SourceRootId;
    /// Contents of the source root.
    #[salsa::input]
    fn source_root(&self, id: SourceRootId) -> Arc<SourceRoot>;

    fn source_root_crates(&self, id: SourceRootId) -> Arc<[CrateId]>;
}

fn source_root_crates(db: &dyn SourceDatabaseExt, id: SourceRootId) -> Arc<[CrateId]> {
    let graph = db.crate_graph();
    let mut crates = graph
        .iter()
        .filter(|&krate| {
            let root_file = graph[krate].root_file_id;
            db.file_source_root(root_file) == id
        })
        .collect::<Vec<_>>();
    crates.sort();
    crates.dedup();
    crates.into_iter().collect()
}

/// Silly workaround for cyclic deps between the traits
pub struct FileLoaderDelegate<T>(pub T);

impl<T: SourceDatabaseExt> FileLoader for FileLoaderDelegate<&'_ T> {
    fn file_text(&self, file_id: FileId) -> Arc<str> {
        SourceDatabaseExt::file_text(self.0, file_id)
    }
    fn resolve_path(&self, path: AnchoredPath<'_>) -> Option<FileId> {
        // FIXME: this *somehow* should be platform agnostic...
        let source_root = self.0.file_source_root(path.anchor);
        let source_root = self.0.source_root(source_root);
        source_root.resolve_path(path)
    }

    fn relevant_crates(&self, file_id: FileId) -> Arc<[CrateId]> {
        let _p = tracing::span!(tracing::Level::INFO, "relevant_crates").entered();
        let source_root = self.0.file_source_root(file_id);
        self.0.source_root_crates(source_root)
    }
}
