//! base_db defines basic database traits. The concrete DB is defined by ide.
// FIXME: Rename this crate, base db is non descriptive
mod change;
mod input;

use std::hash::BuildHasherDefault;

pub use crate::{
    change::FileChange,
    input::{
        BuiltCrateData, BuiltDependency, Crate, CrateBuilder, CrateBuilderId, CrateDataBuilder,
        CrateDisplayName, CrateGraphBuilder, CrateName, CrateOrigin, CratesIdMap, CratesMap,
        DependencyBuilder, Env, ExtraCrateData, LangCrateOrigin, ProcMacroPaths, ReleaseChannel,
        SourceRoot, SourceRootId, TargetLayoutLoadResult, UniqueCrateData,
    },
};
use dashmap::{DashMap, mapref::entry::Entry};
pub use query_group::{self};
use rustc_hash::{FxHashSet, FxHasher};
pub use salsa::{self};
use salsa::{Durability, Setter};
pub use semver::{BuildMetadata, Prerelease, Version, VersionReq};
use syntax::{Parse, SyntaxError, ast};
use triomphe::Arc;
pub use vfs::{AnchoredPath, AnchoredPathBuf, FileId, VfsPath, file_set::FileSet};

#[macro_export]
macro_rules! impl_intern_key {
    ($id:ident, $loc:ident) => {
        #[salsa::interned(no_debug, no_lifetime)]
        pub struct $id {
            pub loc: $loc,
        }

        // If we derive this salsa prints the values recursively, and this causes us to blow.
        impl ::std::fmt::Debug for $id {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.debug_tuple(stringify!($id))
                    .field(&format_args!("{:04x}", self.0.as_u32()))
                    .finish()
            }
        }
    };
}

pub trait Upcast<T: ?Sized> {
    fn upcast(&self) -> &T;
}

pub const DEFAULT_FILE_TEXT_LRU_CAP: u16 = 16;
pub const DEFAULT_PARSE_LRU_CAP: u16 = 128;
pub const DEFAULT_BORROWCK_LRU_CAP: u16 = 2024;

#[derive(Debug, Default)]
pub struct Files {
    files: Arc<DashMap<vfs::FileId, FileText, BuildHasherDefault<FxHasher>>>,
    source_roots: Arc<DashMap<SourceRootId, SourceRootInput, BuildHasherDefault<FxHasher>>>,
    file_source_roots: Arc<DashMap<vfs::FileId, FileSourceRootInput, BuildHasherDefault<FxHasher>>>,
}

impl Files {
    pub fn file_text(&self, file_id: vfs::FileId) -> FileText {
        *self.files.get(&file_id).expect("Unable to fetch file; this is a bug")
    }

    pub fn set_file_text(&self, db: &mut dyn SourceDatabase, file_id: vfs::FileId, text: &str) {
        match self.files.entry(file_id) {
            Entry::Occupied(mut occupied) => {
                occupied.get_mut().set_text(db).to(Arc::from(text));
            }
            Entry::Vacant(vacant) => {
                let text = FileText::new(db, Arc::from(text), file_id);
                vacant.insert(text);
            }
        };
    }

    pub fn set_file_text_with_durability(
        &self,
        db: &mut dyn SourceDatabase,
        file_id: vfs::FileId,
        text: &str,
        durability: Durability,
    ) {
        match self.files.entry(file_id) {
            Entry::Occupied(mut occupied) => {
                occupied.get_mut().set_text(db).with_durability(durability).to(Arc::from(text));
            }
            Entry::Vacant(vacant) => {
                let text =
                    FileText::builder(Arc::from(text), file_id).durability(durability).new(db);
                vacant.insert(text);
            }
        };
    }

    /// Source root of the file.
    pub fn source_root(&self, source_root_id: SourceRootId) -> SourceRootInput {
        let source_root = self
            .source_roots
            .get(&source_root_id)
            .expect("Unable to fetch source root id; this is a bug");

        *source_root
    }

    pub fn set_source_root_with_durability(
        &self,
        db: &mut dyn SourceDatabase,
        source_root_id: SourceRootId,
        source_root: Arc<SourceRoot>,
        durability: Durability,
    ) {
        match self.source_roots.entry(source_root_id) {
            Entry::Occupied(mut occupied) => {
                occupied.get_mut().set_source_root(db).with_durability(durability).to(source_root);
            }
            Entry::Vacant(vacant) => {
                let source_root =
                    SourceRootInput::builder(source_root).durability(durability).new(db);
                vacant.insert(source_root);
            }
        };
    }

    pub fn file_source_root(&self, id: vfs::FileId) -> FileSourceRootInput {
        let file_source_root = self
            .file_source_roots
            .get(&id)
            .expect("Unable to fetch FileSourceRootInput; this is a bug");
        *file_source_root
    }

    pub fn set_file_source_root_with_durability(
        &self,
        db: &mut dyn SourceDatabase,
        id: vfs::FileId,
        source_root_id: SourceRootId,
        durability: Durability,
    ) {
        match self.file_source_roots.entry(id) {
            Entry::Occupied(mut occupied) => {
                occupied
                    .get_mut()
                    .set_source_root_id(db)
                    .with_durability(durability)
                    .to(source_root_id);
            }
            Entry::Vacant(vacant) => {
                let file_source_root =
                    FileSourceRootInput::builder(source_root_id).durability(durability).new(db);
                vacant.insert(file_source_root);
            }
        };
    }
}

#[salsa::interned(no_lifetime)]
pub struct EditionedFileId {
    pub editioned_file_id: span::EditionedFileId,
}

impl EditionedFileId {
    pub fn file_id(&self, db: &dyn salsa::Database) -> vfs::FileId {
        let id = self.editioned_file_id(db);
        id.file_id()
    }

    fn unpack(&self, db: &dyn salsa::Database) -> (vfs::FileId, span::Edition) {
        let id = self.editioned_file_id(db);
        (id.file_id(), id.edition())
    }
}

#[salsa::input]
pub struct FileText {
    pub text: Arc<str>,
    pub file_id: vfs::FileId,
}

#[salsa::input]
pub struct FileSourceRootInput {
    pub source_root_id: SourceRootId,
}

#[salsa::input]
pub struct SourceRootInput {
    pub source_root: Arc<SourceRoot>,
}

/// Database which stores all significant input facts: source code and project
/// model. Everything else in rust-analyzer is derived from these queries.
#[query_group::query_group]
pub trait RootQueryDb: SourceDatabase + salsa::Database {
    /// Parses the file into the syntax tree.
    #[salsa::invoke_actual(parse)]
    #[salsa::lru(128)]
    fn parse(&self, file_id: EditionedFileId) -> Parse<ast::SourceFile>;

    /// Returns the set of errors obtained from parsing the file including validation errors.
    #[salsa::transparent]
    fn parse_errors(&self, file_id: EditionedFileId) -> Option<&[SyntaxError]>;

    #[salsa::transparent]
    fn toolchain_channel(&self, krate: Crate) -> Option<ReleaseChannel>;

    /// Crates whose root file is in `id`.
    fn source_root_crates(&self, id: SourceRootId) -> Arc<[Crate]>;

    #[salsa::transparent]
    fn relevant_crates(&self, file_id: FileId) -> Arc<[Crate]>;

    /// Returns the crates in topological order.
    ///
    /// **Warning**: do not use this query in `hir-*` crates! It kills incrementality across crate metadata modifications.
    #[salsa::input]
    fn all_crates(&self) -> Arc<Box<[Crate]>>;

    /// Returns an iterator over all transitive dependencies of the given crate,
    /// including the crate itself.
    ///
    /// **Warning**: do not use this query in `hir-*` crates! It kills incrementality across crate metadata modifications.
    #[salsa::transparent]
    fn transitive_deps(&self, crate_id: Crate) -> FxHashSet<Crate>;

    /// Returns all transitive reverse dependencies of the given crate,
    /// including the crate itself.
    ///
    /// **Warning**: do not use this query in `hir-*` crates! It kills incrementality across crate metadata modifications.
    #[salsa::invoke(input::transitive_rev_deps)]
    #[salsa::transparent]
    fn transitive_rev_deps(&self, of: Crate) -> FxHashSet<Crate>;
}

pub fn transitive_deps(db: &dyn SourceDatabase, crate_id: Crate) -> FxHashSet<Crate> {
    // There is a bit of duplication here and in `CrateGraphBuilder` in the same method, but it's not terrible
    // and removing that is a bit difficult.
    let mut worklist = vec![crate_id];
    let mut deps = FxHashSet::default();

    while let Some(krate) = worklist.pop() {
        if !deps.insert(krate) {
            continue;
        }

        worklist.extend(krate.data(db).dependencies.iter().map(|dep| dep.crate_id));
    }

    deps
}

#[salsa::db]
pub trait SourceDatabase: salsa::Database {
    /// Text of the file.
    fn file_text(&self, file_id: vfs::FileId) -> FileText;

    fn set_file_text(&mut self, file_id: vfs::FileId, text: &str);

    fn set_file_text_with_durability(
        &mut self,
        file_id: vfs::FileId,
        text: &str,
        durability: Durability,
    );

    /// Contents of the source root.
    fn source_root(&self, id: SourceRootId) -> SourceRootInput;

    fn file_source_root(&self, id: vfs::FileId) -> FileSourceRootInput;

    fn set_file_source_root_with_durability(
        &mut self,
        id: vfs::FileId,
        source_root_id: SourceRootId,
        durability: Durability,
    );

    /// Source root of the file.
    fn set_source_root_with_durability(
        &mut self,
        source_root_id: SourceRootId,
        source_root: Arc<SourceRoot>,
        durability: Durability,
    );

    fn resolve_path(&self, path: AnchoredPath<'_>) -> Option<FileId> {
        // FIXME: this *somehow* should be platform agnostic...
        let source_root = self.file_source_root(path.anchor);
        let source_root = self.source_root(source_root.source_root_id(self));
        source_root.source_root(self).resolve_path(path)
    }

    #[doc(hidden)]
    fn crates_map(&self) -> Arc<CratesMap>;
}

/// Crate related data shared by the whole workspace.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct CrateWorkspaceData {
    // FIXME: Consider removing this, making HirDatabase::target_data_layout an input query
    pub data_layout: TargetLayoutLoadResult,
    /// Toolchain version used to compile the crate.
    pub toolchain: Option<Version>,
}

fn toolchain_channel(db: &dyn RootQueryDb, krate: Crate) -> Option<ReleaseChannel> {
    krate.workspace_data(db).toolchain.as_ref().and_then(|v| ReleaseChannel::from_str(&v.pre))
}

fn parse(db: &dyn RootQueryDb, file_id: EditionedFileId) -> Parse<ast::SourceFile> {
    let _p = tracing::info_span!("parse", ?file_id).entered();
    let (file_id, edition) = file_id.unpack(db.as_dyn_database());
    let text = db.file_text(file_id).text(db);
    ast::SourceFile::parse(&text, edition)
}

fn parse_errors(db: &dyn RootQueryDb, file_id: EditionedFileId) -> Option<&[SyntaxError]> {
    #[salsa::tracked(return_ref)]
    fn parse_errors(db: &dyn RootQueryDb, file_id: EditionedFileId) -> Option<Box<[SyntaxError]>> {
        let errors = db.parse(file_id).errors();
        match &*errors {
            [] => None,
            [..] => Some(errors.into()),
        }
    }
    parse_errors(db, file_id).as_ref().map(|it| &**it)
}

fn source_root_crates(db: &dyn RootQueryDb, id: SourceRootId) -> Arc<[Crate]> {
    let crates = db.all_crates();
    crates
        .iter()
        .copied()
        .filter(|&krate| {
            let root_file = krate.data(db).root_file_id;
            db.file_source_root(root_file).source_root_id(db) == id
        })
        .collect()
}

fn relevant_crates(db: &dyn RootQueryDb, file_id: FileId) -> Arc<[Crate]> {
    let _p = tracing::info_span!("relevant_crates").entered();

    let source_root = db.file_source_root(file_id);
    db.source_root_crates(source_root.source_root_id(db))
}
