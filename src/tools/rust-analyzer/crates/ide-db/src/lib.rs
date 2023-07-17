//! This crate defines the core datastructure representing IDE state -- `RootDatabase`.
//!
//! It is mainly a `HirDatabase` for semantic analysis, plus a `SymbolsDatabase`, for fuzzy search.

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

mod apply_change;

pub mod active_parameter;
pub mod assists;
pub mod defs;
pub mod famous_defs;
pub mod helpers;
pub mod items_locator;
pub mod label;
pub mod path_transform;
pub mod rename;
pub mod rust_doc;
pub mod search;
pub mod source_change;
pub mod symbol_index;
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
    pub mod node_ext;
    pub mod insert_whitespace_into_node;
    pub mod format_string;
    pub mod format_string_exprs;

    pub use parser::LexedStr;
}

use std::{fmt, mem::ManuallyDrop};

use base_db::{
    salsa::{self, Durability},
    AnchoredPath, CrateId, FileId, FileLoader, FileLoaderDelegate, SourceDatabase, Upcast,
};
use hir::db::{DefDatabase, ExpandDatabase, HirDatabase};
use triomphe::Arc;

use crate::{line_index::LineIndex, symbol_index::SymbolsDatabase};
pub use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

pub use ::line_index;

/// `base_db` is normally also needed in places where `ide_db` is used, so this re-export is for convenience.
pub use base_db;

pub type FxIndexSet<T> = indexmap::IndexSet<T, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
pub type FxIndexMap<K, V> =
    indexmap::IndexMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

#[salsa::database(
    base_db::SourceDatabaseExtStorage,
    base_db::SourceDatabaseStorage,
    hir::db::ExpandDatabaseStorage,
    hir::db::DefDatabaseStorage,
    hir::db::HirDatabaseStorage,
    hir::db::InternDatabaseStorage,
    LineIndexDatabaseStorage,
    symbol_index::SymbolsDatabaseStorage
)]
pub struct RootDatabase {
    // We use `ManuallyDrop` here because every codegen unit that contains a
    // `&RootDatabase -> &dyn OtherDatabase` cast will instantiate its drop glue in the vtable,
    // which duplicates `Weak::drop` and `Arc::drop` tens of thousands of times, which makes
    // compile times of all `ide_*` and downstream crates suffer greatly.
    storage: ManuallyDrop<salsa::Storage<RootDatabase>>,
}

impl Drop for RootDatabase {
    fn drop(&mut self) {
        unsafe { ManuallyDrop::drop(&mut self.storage) };
    }
}

impl fmt::Debug for RootDatabase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RootDatabase").finish()
    }
}

impl Upcast<dyn ExpandDatabase> for RootDatabase {
    fn upcast(&self) -> &(dyn ExpandDatabase + 'static) {
        &*self
    }
}

impl Upcast<dyn DefDatabase> for RootDatabase {
    fn upcast(&self) -> &(dyn DefDatabase + 'static) {
        &*self
    }
}

impl Upcast<dyn HirDatabase> for RootDatabase {
    fn upcast(&self) -> &(dyn HirDatabase + 'static) {
        &*self
    }
}

impl FileLoader for RootDatabase {
    fn file_text(&self, file_id: FileId) -> Arc<str> {
        FileLoaderDelegate(self).file_text(file_id)
    }
    fn resolve_path(&self, path: AnchoredPath<'_>) -> Option<FileId> {
        FileLoaderDelegate(self).resolve_path(path)
    }
    fn relevant_crates(&self, file_id: FileId) -> Arc<FxHashSet<CrateId>> {
        FileLoaderDelegate(self).relevant_crates(file_id)
    }
}

impl salsa::Database for RootDatabase {}

impl Default for RootDatabase {
    fn default() -> RootDatabase {
        RootDatabase::new(None)
    }
}

impl RootDatabase {
    pub fn new(lru_capacity: Option<usize>) -> RootDatabase {
        let mut db = RootDatabase { storage: ManuallyDrop::new(salsa::Storage::default()) };
        db.set_crate_graph_with_durability(Default::default(), Durability::HIGH);
        db.set_proc_macros_with_durability(Default::default(), Durability::HIGH);
        db.set_local_roots_with_durability(Default::default(), Durability::HIGH);
        db.set_library_roots_with_durability(Default::default(), Durability::HIGH);
        db.set_expand_proc_attr_macros_with_durability(false, Durability::HIGH);
        db.update_parse_query_lru_capacity(lru_capacity);
        db
    }

    pub fn enable_proc_attr_macros(&mut self) {
        self.set_expand_proc_attr_macros_with_durability(true, Durability::HIGH);
    }

    pub fn update_parse_query_lru_capacity(&mut self, lru_capacity: Option<usize>) {
        let lru_capacity = lru_capacity.unwrap_or(base_db::DEFAULT_PARSE_LRU_CAP);
        base_db::ParseQuery.in_db_mut(self).set_lru_capacity(lru_capacity);
        // macro expansions are usually rather small, so we can afford to keep more of them alive
        hir::db::ParseMacroExpansionQuery.in_db_mut(self).set_lru_capacity(4 * lru_capacity);
        hir::db::MacroExpandQuery.in_db_mut(self).set_lru_capacity(4 * lru_capacity);
    }

    pub fn update_lru_capacities(&mut self, lru_capacities: &FxHashMap<Box<str>, usize>) {
        use hir::db as hir_db;

        base_db::ParseQuery.in_db_mut(self).set_lru_capacity(
            lru_capacities
                .get(stringify!(ParseQuery))
                .copied()
                .unwrap_or(base_db::DEFAULT_PARSE_LRU_CAP),
        );
        hir_db::ParseMacroExpansionQuery.in_db_mut(self).set_lru_capacity(
            lru_capacities
                .get(stringify!(ParseMacroExpansionQuery))
                .copied()
                .unwrap_or(4 * base_db::DEFAULT_PARSE_LRU_CAP),
        );
        hir_db::MacroExpandQuery.in_db_mut(self).set_lru_capacity(
            lru_capacities
                .get(stringify!(MacroExpandQuery))
                .copied()
                .unwrap_or(4 * base_db::DEFAULT_PARSE_LRU_CAP),
        );

        macro_rules! update_lru_capacity_per_query {
            ($( $module:ident :: $query:ident )*) => {$(
                if let Some(&cap) = lru_capacities.get(stringify!($query)) {
                    $module::$query.in_db_mut(self).set_lru_capacity(cap);
                }
            )*}
        }
        update_lru_capacity_per_query![
            // SourceDatabase
            // base_db::ParseQuery
            // base_db::CrateGraphQuery
            // base_db::ProcMacrosQuery

            // SourceDatabaseExt
            // base_db::FileTextQuery
            // base_db::FileSourceRootQuery
            // base_db::SourceRootQuery
            base_db::SourceRootCratesQuery

            // ExpandDatabase
            hir_db::AstIdMapQuery
            // hir_db::ParseMacroExpansionQuery
            // hir_db::InternMacroCallQuery
            hir_db::MacroArgNodeQuery
            hir_db::DeclMacroExpanderQuery
            // hir_db::MacroExpandQuery
            hir_db::ExpandProcMacroQuery
            hir_db::HygieneFrameQuery
            hir_db::ParseMacroExpansionErrorQuery

            // DefDatabase
            hir_db::FileItemTreeQuery
            hir_db::CrateDefMapQueryQuery
            hir_db::BlockDefMapQuery
            hir_db::StructDataQuery
            hir_db::StructDataWithDiagnosticsQuery
            hir_db::UnionDataQuery
            hir_db::UnionDataWithDiagnosticsQuery
            hir_db::EnumDataQuery
            hir_db::EnumDataWithDiagnosticsQuery
            hir_db::ImplDataQuery
            hir_db::ImplDataWithDiagnosticsQuery
            hir_db::TraitDataQuery
            hir_db::TraitDataWithDiagnosticsQuery
            hir_db::TraitAliasDataQuery
            hir_db::TypeAliasDataQuery
            hir_db::FunctionDataQuery
            hir_db::ConstDataQuery
            hir_db::StaticDataQuery
            hir_db::Macro2DataQuery
            hir_db::MacroRulesDataQuery
            hir_db::ProcMacroDataQuery
            hir_db::BodyWithSourceMapQuery
            hir_db::BodyQuery
            hir_db::ExprScopesQuery
            hir_db::GenericParamsQuery
            hir_db::VariantsAttrsQuery
            hir_db::FieldsAttrsQuery
            hir_db::VariantsAttrsSourceMapQuery
            hir_db::FieldsAttrsSourceMapQuery
            hir_db::AttrsQuery
            hir_db::CrateLangItemsQuery
            hir_db::LangItemQuery
            hir_db::ImportMapQuery
            hir_db::FieldVisibilitiesQuery
            hir_db::FunctionVisibilityQuery
            hir_db::ConstVisibilityQuery
            hir_db::CrateSupportsNoStdQuery

            // HirDatabase
            hir_db::InferQueryQuery
            hir_db::MirBodyQuery
            hir_db::BorrowckQuery
            hir_db::TyQuery
            hir_db::ValueTyQuery
            hir_db::ImplSelfTyQuery
            hir_db::ConstParamTyQuery
            hir_db::ConstEvalQuery
            hir_db::ConstEvalDiscriminantQuery
            hir_db::ImplTraitQuery
            hir_db::FieldTypesQuery
            hir_db::LayoutOfAdtQuery
            hir_db::TargetDataLayoutQuery
            hir_db::CallableItemSignatureQuery
            hir_db::ReturnTypeImplTraitsQuery
            hir_db::GenericPredicatesForParamQuery
            hir_db::GenericPredicatesQuery
            hir_db::TraitEnvironmentQuery
            hir_db::GenericDefaultsQuery
            hir_db::InherentImplsInCrateQuery
            hir_db::InherentImplsInBlockQuery
            hir_db::IncoherentInherentImplCratesQuery
            hir_db::TraitImplsInCrateQuery
            hir_db::TraitImplsInBlockQuery
            hir_db::TraitImplsInDepsQuery
            // hir_db::InternCallableDefQuery
            // hir_db::InternLifetimeParamIdQuery
            // hir_db::InternImplTraitIdQuery
            // hir_db::InternTypeOrConstParamIdQuery
            // hir_db::InternClosureQuery
            // hir_db::InternGeneratorQuery
            hir_db::AssociatedTyDataQuery
            hir_db::TraitDatumQuery
            hir_db::StructDatumQuery
            hir_db::ImplDatumQuery
            hir_db::FnDefDatumQuery
            hir_db::FnDefVarianceQuery
            hir_db::AdtVarianceQuery
            hir_db::AssociatedTyValueQuery
            hir_db::TraitSolveQueryQuery
            hir_db::ProgramClausesForChalkEnvQuery

            // SymbolsDatabase
            symbol_index::ModuleSymbolsQuery
            symbol_index::LibrarySymbolsQuery
            // symbol_index::LocalRootsQuery
            // symbol_index::LibraryRootsQuery

            // LineIndexDatabase
            crate::LineIndexQuery

            // InternDatabase
            // hir_db::InternFunctionQuery
            // hir_db::InternStructQuery
            // hir_db::InternUnionQuery
            // hir_db::InternEnumQuery
            // hir_db::InternConstQuery
            // hir_db::InternStaticQuery
            // hir_db::InternTraitQuery
            // hir_db::InternTraitAliasQuery
            // hir_db::InternTypeAliasQuery
            // hir_db::InternImplQuery
            // hir_db::InternExternBlockQuery
            // hir_db::InternBlockQuery
            // hir_db::InternMacro2Query
            // hir_db::InternProcMacroQuery
            // hir_db::InternMacroRulesQuery
        ];
    }
}

impl salsa::ParallelDatabase for RootDatabase {
    fn snapshot(&self) -> salsa::Snapshot<RootDatabase> {
        salsa::Snapshot::new(RootDatabase { storage: ManuallyDrop::new(self.storage.snapshot()) })
    }
}

#[salsa::query_group(LineIndexDatabaseStorage)]
pub trait LineIndexDatabase: base_db::SourceDatabase {
    fn line_index(&self, file_id: FileId) -> Arc<LineIndex>;
}

fn line_index(db: &dyn LineIndexDatabase, file_id: FileId) -> Arc<LineIndex> {
    let text = db.file_text(file_id);
    Arc::new(LineIndex::new(&text))
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
    Impl,
    Label,
    LifetimeParam,
    Local,
    Macro,
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
            hir::MacroKind::Declarative | hir::MacroKind::BuiltIn | hir::MacroKind::ProcMacro => {
                SymbolKind::Macro
            }
            hir::MacroKind::Derive => SymbolKind::Derive,
            hir::MacroKind::Attr => SymbolKind::Attribute,
        }
    }
}

impl From<hir::ModuleDefId> for SymbolKind {
    fn from(it: hir::ModuleDefId) -> Self {
        match it {
            hir::ModuleDefId::ConstId(..) => SymbolKind::Const,
            hir::ModuleDefId::EnumVariantId(..) => SymbolKind::Variant,
            hir::ModuleDefId::FunctionId(..) => SymbolKind::Function,
            hir::ModuleDefId::MacroId(..) => SymbolKind::Macro,
            hir::ModuleDefId::ModuleId(..) => SymbolKind::Module,
            hir::ModuleDefId::StaticId(..) => SymbolKind::Static,
            hir::ModuleDefId::AdtId(hir::AdtId::StructId(..)) => SymbolKind::Struct,
            hir::ModuleDefId::AdtId(hir::AdtId::EnumId(..)) => SymbolKind::Enum,
            hir::ModuleDefId::AdtId(hir::AdtId::UnionId(..)) => SymbolKind::Union,
            hir::ModuleDefId::TraitId(..) => SymbolKind::Trait,
            hir::ModuleDefId::TraitAliasId(..) => SymbolKind::TraitAlias,
            hir::ModuleDefId::TypeAliasId(..) => SymbolKind::TypeAlias,
            hir::ModuleDefId::BuiltinType(..) => SymbolKind::TypeAlias,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnippetCap {
    _private: (),
}

impl SnippetCap {
    pub const fn new(allow_snippets: bool) -> Option<SnippetCap> {
        if allow_snippets {
            Some(SnippetCap { _private: () })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    mod sourcegen_lints;
    mod line_index;
}
