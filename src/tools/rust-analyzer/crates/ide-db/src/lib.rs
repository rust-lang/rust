//! This crate defines the core data structure representing IDE state -- `RootDatabase`.
//!
//! It is mainly a `HirDatabase` for semantic analysis, plus a `SymbolsDatabase`, for fuzzy search.

mod apply_change;

pub mod active_parameter;
pub mod assists;
pub mod defs;
pub mod documentation;
pub mod famous_defs;
pub mod helpers;
pub mod items_locator;
pub mod label;
pub mod path_transform;
pub mod prime_caches;
pub mod rename;
pub mod rust_doc;
pub mod search;
pub mod source_change;
pub mod symbol_index;
pub mod text_edit;
pub mod traits;
pub mod ty_filter;
pub mod use_trivial_constructor;

pub mod imports {
    pub mod import_assets;
    pub mod insert_use;
    pub mod merge_imports;
}

pub mod generated {
    pub mod lints;
}

pub mod syntax_helpers {
    pub mod format_string;
    pub mod format_string_exprs;
    pub mod tree_diff;
    pub use hir::prettify_macro_expansion;
    pub mod node_ext;
    pub mod suggest_name;

    pub use parser::LexedStr;
}

pub use hir::{ChangeWithProcMacros, EditionedFileId};
use salsa::Durability;

use std::{fmt, mem::ManuallyDrop};

use base_db::{
    CrateGraphBuilder, CratesMap, FileSourceRootInput, FileText, Files, RootQueryDb,
    SourceDatabase, SourceRoot, SourceRootId, SourceRootInput, query_group,
};
use hir::{
    FilePositionWrapper, FileRangeWrapper,
    db::{DefDatabase, ExpandDatabase},
};
use triomphe::Arc;

use crate::{line_index::LineIndex, symbol_index::SymbolsDatabase};
pub use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

pub use ::line_index;

/// `base_db` is normally also needed in places where `ide_db` is used, so this re-export is for convenience.
pub use base_db;
pub use span::{self, FileId};

pub type FxIndexSet<T> = indexmap::IndexSet<T, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
pub type FxIndexMap<K, V> =
    indexmap::IndexMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

pub type FilePosition = FilePositionWrapper<FileId>;
pub type FileRange = FileRangeWrapper<FileId>;

#[salsa_macros::db]
pub struct RootDatabase {
    // FIXME: Revisit this commit now that we migrated to the new salsa, given we store arcs in this
    // db directly now
    // We use `ManuallyDrop` here because every codegen unit that contains a
    // `&RootDatabase -> &dyn OtherDatabase` cast will instantiate its drop glue in the vtable,
    // which duplicates `Weak::drop` and `Arc::drop` tens of thousands of times, which makes
    // compile times of all `ide_*` and downstream crates suffer greatly.
    storage: ManuallyDrop<salsa::Storage<Self>>,
    files: Arc<Files>,
    crates_map: Arc<CratesMap>,
}

impl std::panic::RefUnwindSafe for RootDatabase {}

#[salsa_macros::db]
impl salsa::Database for RootDatabase {}

impl Drop for RootDatabase {
    fn drop(&mut self) {
        unsafe { ManuallyDrop::drop(&mut self.storage) };
    }
}

impl Clone for RootDatabase {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            files: self.files.clone(),
            crates_map: self.crates_map.clone(),
        }
    }
}

impl fmt::Debug for RootDatabase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RootDatabase").finish()
    }
}

#[salsa_macros::db]
impl SourceDatabase for RootDatabase {
    fn file_text(&self, file_id: vfs::FileId) -> FileText {
        self.files.file_text(file_id)
    }

    fn set_file_text(&mut self, file_id: vfs::FileId, text: &str) {
        let files = Arc::clone(&self.files);
        files.set_file_text(self, file_id, text);
    }

    fn set_file_text_with_durability(
        &mut self,
        file_id: vfs::FileId,
        text: &str,
        durability: Durability,
    ) {
        let files = Arc::clone(&self.files);
        files.set_file_text_with_durability(self, file_id, text, durability);
    }

    /// Source root of the file.
    fn source_root(&self, source_root_id: SourceRootId) -> SourceRootInput {
        self.files.source_root(source_root_id)
    }

    fn set_source_root_with_durability(
        &mut self,
        source_root_id: SourceRootId,
        source_root: Arc<SourceRoot>,
        durability: Durability,
    ) {
        let files = Arc::clone(&self.files);
        files.set_source_root_with_durability(self, source_root_id, source_root, durability);
    }

    fn file_source_root(&self, id: vfs::FileId) -> FileSourceRootInput {
        self.files.file_source_root(id)
    }

    fn set_file_source_root_with_durability(
        &mut self,
        id: vfs::FileId,
        source_root_id: SourceRootId,
        durability: Durability,
    ) {
        let files = Arc::clone(&self.files);
        files.set_file_source_root_with_durability(self, id, source_root_id, durability);
    }

    fn crates_map(&self) -> Arc<CratesMap> {
        self.crates_map.clone()
    }
}

impl Default for RootDatabase {
    fn default() -> RootDatabase {
        RootDatabase::new(None)
    }
}

impl RootDatabase {
    pub fn new(lru_capacity: Option<u16>) -> RootDatabase {
        let mut db = RootDatabase {
            storage: ManuallyDrop::new(salsa::Storage::default()),
            files: Default::default(),
            crates_map: Default::default(),
        };
        // This needs to be here otherwise `CrateGraphBuilder` will panic.
        db.set_all_crates(Arc::new(Box::new([])));
        CrateGraphBuilder::default().set_in_db(&mut db);
        db.set_proc_macros_with_durability(Default::default(), Durability::MEDIUM);
        db.set_local_roots_with_durability(Default::default(), Durability::MEDIUM);
        db.set_library_roots_with_durability(Default::default(), Durability::MEDIUM);
        db.set_expand_proc_attr_macros_with_durability(false, Durability::HIGH);
        db.update_base_query_lru_capacities(lru_capacity);
        db
    }

    pub fn enable_proc_attr_macros(&mut self) {
        self.set_expand_proc_attr_macros_with_durability(true, Durability::HIGH);
    }

    pub fn update_base_query_lru_capacities(&mut self, _lru_capacity: Option<u16>) {
        // let lru_capacity = lru_capacity.unwrap_or(base_db::DEFAULT_PARSE_LRU_CAP);
        // base_db::FileTextQuery.in_db_mut(self).set_lru_capacity(DEFAULT_FILE_TEXT_LRU_CAP);
        // base_db::ParseQuery.in_db_mut(self).set_lru_capacity(lru_capacity);
        // // macro expansions are usually rather small, so we can afford to keep more of them alive
        // hir::db::ParseMacroExpansionQuery.in_db_mut(self).set_lru_capacity(4 * lru_capacity);
        // hir::db::BorrowckQuery.in_db_mut(self).set_lru_capacity(base_db::DEFAULT_BORROWCK_LRU_CAP);
        // hir::db::BodyWithSourceMapQuery.in_db_mut(self).set_lru_capacity(2048);
    }

    pub fn update_lru_capacities(&mut self, _lru_capacities: &FxHashMap<Box<str>, u16>) {
        // FIXME(salsa-transition): bring this back; allow changing LRU settings at runtime.
        // use hir::db as hir_db;

        // base_db::FileTextQuery.in_db_mut(self).set_lru_capacity(DEFAULT_FILE_TEXT_LRU_CAP);
        // base_db::ParseQuery.in_db_mut(self).set_lru_capacity(
        //     lru_capacities
        //         .get(stringify!(ParseQuery))
        //         .copied()
        //         .unwrap_or(base_db::DEFAULT_PARSE_LRU_CAP),
        // );
        // hir_db::ParseMacroExpansionQuery.in_db_mut(self).set_lru_capacity(
        //     lru_capacities
        //         .get(stringify!(ParseMacroExpansionQuery))
        //         .copied()
        //         .unwrap_or(4 * base_db::DEFAULT_PARSE_LRU_CAP),
        // );
        // hir_db::BorrowckQuery.in_db_mut(self).set_lru_capacity(
        //     lru_capacities
        //         .get(stringify!(BorrowckQuery))
        //         .copied()
        //         .unwrap_or(base_db::DEFAULT_BORROWCK_LRU_CAP),
        // );
        // hir::db::BodyWithSourceMapQuery.in_db_mut(self).set_lru_capacity(2048);
    }
}

#[query_group::query_group]
pub trait LineIndexDatabase: base_db::RootQueryDb {
    #[salsa::invoke_interned(line_index)]
    fn line_index(&self, file_id: FileId) -> Arc<LineIndex>;
}

fn line_index(db: &dyn LineIndexDatabase, file_id: FileId) -> Arc<LineIndex> {
    let text = db.file_text(file_id).text(db);
    Arc::new(LineIndex::new(text))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SymbolKind {
    Attribute,
    BuiltinAttr,
    Const,
    ConstParam,
    Derive,
    DeriveHelper,
    Enum,
    Field,
    Function,
    Method,
    Impl,
    InlineAsmRegOrRegClass,
    Label,
    LifetimeParam,
    Local,
    Macro,
    ProcMacro,
    Module,
    SelfParam,
    SelfType,
    Static,
    Struct,
    ToolModule,
    Trait,
    TraitAlias,
    TypeAlias,
    TypeParam,
    Union,
    ValueParam,
    Variant,
}

impl From<hir::MacroKind> for SymbolKind {
    fn from(it: hir::MacroKind) -> Self {
        match it {
            hir::MacroKind::Declarative | hir::MacroKind::DeclarativeBuiltIn => SymbolKind::Macro,
            hir::MacroKind::ProcMacro => SymbolKind::ProcMacro,
            hir::MacroKind::Derive | hir::MacroKind::DeriveBuiltIn => SymbolKind::Derive,
            hir::MacroKind::Attr | hir::MacroKind::AttrBuiltIn => SymbolKind::Attribute,
        }
    }
}

impl From<hir::ModuleDef> for SymbolKind {
    fn from(it: hir::ModuleDef) -> Self {
        match it {
            hir::ModuleDef::Const(..) => SymbolKind::Const,
            hir::ModuleDef::Variant(..) => SymbolKind::Variant,
            hir::ModuleDef::Function(..) => SymbolKind::Function,
            hir::ModuleDef::Macro(mac) if mac.is_proc_macro() => SymbolKind::ProcMacro,
            hir::ModuleDef::Macro(..) => SymbolKind::Macro,
            hir::ModuleDef::Module(..) => SymbolKind::Module,
            hir::ModuleDef::Static(..) => SymbolKind::Static,
            hir::ModuleDef::Adt(hir::Adt::Struct(..)) => SymbolKind::Struct,
            hir::ModuleDef::Adt(hir::Adt::Enum(..)) => SymbolKind::Enum,
            hir::ModuleDef::Adt(hir::Adt::Union(..)) => SymbolKind::Union,
            hir::ModuleDef::Trait(..) => SymbolKind::Trait,
            hir::ModuleDef::TraitAlias(..) => SymbolKind::TraitAlias,
            hir::ModuleDef::TypeAlias(..) => SymbolKind::TypeAlias,
            hir::ModuleDef::BuiltinType(..) => SymbolKind::TypeAlias,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnippetCap {
    _private: (),
}

impl SnippetCap {
    pub const fn new(allow_snippets: bool) -> Option<SnippetCap> {
        if allow_snippets { Some(SnippetCap { _private: () }) } else { None }
    }
}

pub struct Ranker<'a> {
    pub kind: parser::SyntaxKind,
    pub text: &'a str,
    pub ident_kind: bool,
}

impl<'a> Ranker<'a> {
    pub const MAX_RANK: usize = 0b1110;

    pub fn from_token(token: &'a syntax::SyntaxToken) -> Self {
        let kind = token.kind();
        Ranker { kind, text: token.text(), ident_kind: kind.is_any_identifier() }
    }

    /// A utility function that ranks a token again a given kind and text, returning a number that
    /// represents how close the token is to the given kind and text.
    pub fn rank_token(&self, tok: &syntax::SyntaxToken) -> usize {
        let tok_kind = tok.kind();

        let exact_same_kind = tok_kind == self.kind;
        let both_idents = exact_same_kind || (tok_kind.is_any_identifier() && self.ident_kind);
        let same_text = tok.text() == self.text;
        // anything that mapped into a token tree has likely no semantic information
        let no_tt_parent =
            tok.parent().is_some_and(|it| it.kind() != parser::SyntaxKind::TOKEN_TREE);
        (both_idents as usize)
            | ((exact_same_kind as usize) << 1)
            | ((same_text as usize) << 2)
            | ((no_tt_parent as usize) << 3)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Severity {
    Error,
    Warning,
    WeakWarning,
    Allow,
}
