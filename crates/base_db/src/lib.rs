//! base_db defines basic database traits. The concrete DB is defined by ide.
mod cancellation;
mod input;
pub mod fixture;

use std::{panic, sync::Arc};

use rustc_hash::FxHashSet;
use syntax::{ast, Parse, SourceFile, TextRange, TextSize};

pub use crate::{
    cancellation::Canceled,
    input::{
        CrateData, CrateGraph, CrateId, CrateName, Dependency, Edition, Env, FileId, ProcMacroId,
        SourceRoot, SourceRootId,
    },
};
pub use salsa;
pub use vfs::{file_set::FileSet, VfsPath};

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
    pub offset: TextSize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FileRange {
    pub file_id: FileId,
    pub range: TextRange,
}

pub const DEFAULT_LRU_CAP: usize = 128;

pub trait FileLoader {
    /// Text of the file.
    fn file_text(&self, file_id: FileId) -> Arc<String>;
    /// Note that we intentionally accept a `&str` and not a `&Path` here. This
    /// method exists to handle `#[path = "/some/path.rs"] mod foo;` and such,
    /// so the input is guaranteed to be utf-8 string. One might be tempted to
    /// introduce some kind of "utf-8 path with / separators", but that's a bad idea. Behold
    /// `#[path = "C://no/way"]`
    fn resolve_path(&self, anchor: FileId, path: &str) -> Option<FileId>;
    fn relevant_crates(&self, file_id: FileId) -> Arc<FxHashSet<CrateId>>;
    fn possible_sudmobule_names(&self, module_file: FileId) -> Vec<String>;
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

    fn source_root_crates(&self, id: SourceRootId) -> Arc<FxHashSet<CrateId>>;
}

fn source_root_crates(db: &dyn SourceDatabaseExt, id: SourceRootId) -> Arc<FxHashSet<CrateId>> {
    let graph = db.crate_graph();
    let res = graph
        .iter()
        .filter(|&krate| {
            let root_file = graph[krate].root_file_id;
            db.file_source_root(root_file) == id
        })
        .collect::<FxHashSet<_>>();
    Arc::new(res)
}

/// Silly workaround for cyclic deps between the traits
pub struct FileLoaderDelegate<T>(pub T);

impl<T: SourceDatabaseExt> FileLoader for FileLoaderDelegate<&'_ T> {
    fn file_text(&self, file_id: FileId) -> Arc<String> {
        SourceDatabaseExt::file_text(self.0, file_id)
    }
    fn resolve_path(&self, anchor: FileId, path: &str) -> Option<FileId> {
        // FIXME: this *somehow* should be platform agnostic...
        // self.source_root(anchor)
        let source_root = self.source_root(anchor);
        source_root.file_set.resolve_path(anchor, path)
    }

    fn relevant_crates(&self, file_id: FileId) -> Arc<FxHashSet<CrateId>> {
        let source_root = self.0.file_source_root(file_id);
        self.0.source_root_crates(source_root)
    }

    fn possible_sudmobule_names(&self, module_file: FileId) -> Vec<String> {
        possible_sudmobule_names(&self.source_root(module_file).file_set, module_file)
    }
}

impl<T: SourceDatabaseExt> FileLoaderDelegate<&'_ T> {
    fn source_root(&self, anchor: FileId) -> Arc<SourceRoot> {
        let source_root = self.0.file_source_root(anchor);
        self.0.source_root(source_root)
    }
}

fn possible_sudmobule_names(module_files: &FileSet, module_file: FileId) -> Vec<String> {
    let directory_to_look_for_submodules = match module_files
        .path_for_file(&module_file)
        .and_then(|module_file_path| get_directory_with_submodules(module_file_path))
    {
        Some(directory) => directory,
        None => return Vec::new(),
    };
    module_files
        .iter()
        .filter(|submodule_file| submodule_file != &module_file)
        .filter_map(|submodule_file| {
            let submodule_path = module_files.path_for_file(&submodule_file)?;
            if submodule_path.parent()? == directory_to_look_for_submodules {
                submodule_path.file_name_and_extension()
            } else {
                None
            }
        })
        .filter_map(|file_name_and_extension| {
            match file_name_and_extension {
                // TODO kb wrong resolution for nested non-file modules (mod tests { mod <|> })
                // TODO kb in src/bin when a module is included into another,
                // the included file gets "moved" into a directory below and now cannot add any other modules
                ("mod", Some("rs")) | ("lib", Some("rs")) | ("main", Some("rs")) => None,
                (file_name, Some("rs")) => Some(file_name.to_owned()),
                (subdirectory_name, None) => {
                    let mod_rs_path =
                        directory_to_look_for_submodules.join(subdirectory_name)?.join("mod.rs")?;
                    if module_files.file_for_path(&mod_rs_path).is_some() {
                        Some(subdirectory_name.to_owned())
                    } else {
                        None
                    }
                }
                _ => None,
            }
        })
        .collect()
}

fn get_directory_with_submodules(module_file_path: &VfsPath) -> Option<VfsPath> {
    let module_directory_path = module_file_path.parent()?;
    match module_file_path.file_name_and_extension()? {
        ("mod", Some("rs")) | ("lib", Some("rs")) | ("main", Some("rs")) => {
            Some(module_directory_path)
        }
        (regular_rust_file_name, Some("rs")) => {
            if matches!(
                (
                    module_directory_path
                        .parent()
                        .as_ref()
                        .and_then(|path| path.file_name_and_extension()),
                    module_directory_path.file_name_and_extension(),
                ),
                (Some(("src", None)), Some(("bin", None)))
            ) {
                // files in /src/bin/ can import each other directly
                Some(module_directory_path)
            } else {
                module_directory_path.join(regular_rust_file_name)
            }
        }
        _ => None,
    }
}
