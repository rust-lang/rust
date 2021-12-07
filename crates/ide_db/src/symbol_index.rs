//! This module handles fuzzy-searching of functions, structs and other symbols
//! by name across the whole workspace and dependencies.
//!
//! It works by building an incrementally-updated text-search index of all
//! symbols. The backbone of the index is the **awesome** `fst` crate by
//! @BurntSushi.
//!
//! In a nutshell, you give a set of strings to `fst`, and it builds a
//! finite state machine describing this set of strings. The strings which
//! could fuzzy-match a pattern can also be described by a finite state machine.
//! What is freaking cool is that you can now traverse both state machines in
//! lock-step to enumerate the strings which are both in the input set and
//! fuzz-match the query. Or, more formally, given two languages described by
//! FSTs, one can build a product FST which describes the intersection of the
//! languages.
//!
//! `fst` does not support cheap updating of the index, but it supports unioning
//! of state machines. So, to account for changing source code, we build an FST
//! for each library (which is assumed to never change) and an FST for each Rust
//! file in the current workspace, and run a query against the union of all
//! those FSTs.

use std::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    mem,
    sync::Arc,
};

use base_db::{
    salsa::{self, ParallelDatabase},
    CrateId, FileRange, SourceDatabaseExt, SourceRootId, Upcast,
};
use either::Either;
use fst::{self, Streamer};
use hir::{
    db::{DefDatabase, HirDatabase},
    AdtId, AssocItemId, AssocItemLoc, DefHasSource, DefWithBodyId, HasSource, HirFileId, ImplId,
    InFile, ItemContainerId, ItemLoc, ItemTreeNode, Lookup, MacroDef, Module, ModuleDefId,
    ModuleId, Semantics, TraitId,
};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use syntax::{ast::HasName, AstNode, SmolStr, SyntaxNode, SyntaxNodePtr};

use crate::{RootDatabase, SymbolKind};

#[derive(Debug)]
pub struct Query {
    query: String,
    lowercased: String,
    only_types: bool,
    libs: bool,
    exact: bool,
    case_sensitive: bool,
    limit: usize,
}

impl Query {
    pub fn new(query: String) -> Query {
        let lowercased = query.to_lowercase();
        Query {
            query,
            lowercased,
            only_types: false,
            libs: false,
            exact: false,
            case_sensitive: false,
            limit: usize::max_value(),
        }
    }

    pub fn only_types(&mut self) {
        self.only_types = true;
    }

    pub fn libs(&mut self) {
        self.libs = true;
    }

    pub fn exact(&mut self) {
        self.exact = true;
    }

    pub fn case_sensitive(&mut self) {
        self.case_sensitive = true;
    }

    pub fn limit(&mut self, limit: usize) {
        self.limit = limit
    }
}

#[salsa::query_group(SymbolsDatabaseStorage)]
pub trait SymbolsDatabase: HirDatabase + SourceDatabaseExt + Upcast<dyn HirDatabase> {
    /// The symbol index for a given module. These modules should only be in source roots that
    /// are inside local_roots.
    fn module_symbols(&self, module_id: ModuleId) -> Arc<SymbolIndex>;

    /// The symbol index for a given source root within library_roots.
    fn library_symbols(&self, source_root_id: SourceRootId) -> Arc<SymbolIndex>;

    /// The set of "local" (that is, from the current workspace) roots.
    /// Files in local roots are assumed to change frequently.
    #[salsa::input]
    fn local_roots(&self) -> Arc<FxHashSet<SourceRootId>>;

    /// The set of roots for crates.io libraries.
    /// Files in libraries are assumed to never change.
    #[salsa::input]
    fn library_roots(&self) -> Arc<FxHashSet<SourceRootId>>;
}

fn library_symbols(db: &dyn SymbolsDatabase, source_root_id: SourceRootId) -> Arc<SymbolIndex> {
    let _p = profile::span("library_symbols");

    // todo: this could be parallelized, once I figure out how to do that...
    let symbols = db
        .source_root_crates(source_root_id)
        .iter()
        .flat_map(|&krate| module_ids_for_crate(db.upcast(), krate))
        // we specifically avoid calling SymbolsDatabase::module_symbols here, even they do the same thing,
        // as the index for a library is not going to really ever change, and we do not want to store each
        // module's index in salsa.
        .map(|module_id| SymbolCollector::collect(db, module_id))
        .flatten()
        .collect();

    Arc::new(SymbolIndex::new(symbols))
}

fn module_symbols(db: &dyn SymbolsDatabase, module_id: ModuleId) -> Arc<SymbolIndex> {
    let _p = profile::span("module_symbols");
    let symbols = SymbolCollector::collect(db, module_id);
    Arc::new(SymbolIndex::new(symbols))
}

/// Need to wrap Snapshot to provide `Clone` impl for `map_with`
struct Snap<DB>(DB);
impl<DB: ParallelDatabase> Snap<salsa::Snapshot<DB>> {
    fn new(db: &DB) -> Self {
        Self(db.snapshot())
    }
}
impl<DB: ParallelDatabase> Clone for Snap<salsa::Snapshot<DB>> {
    fn clone(&self) -> Snap<salsa::Snapshot<DB>> {
        Snap(self.0.snapshot())
    }
}
impl<DB> std::ops::Deref for Snap<DB> {
    type Target = DB;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Feature: Workspace Symbol
//
// Uses fuzzy-search to find types, modules and functions by name across your
// project and dependencies. This is **the** most useful feature, which improves code
// navigation tremendously. It mostly works on top of the built-in LSP
// functionality, however `#` and `*` symbols can be used to narrow down the
// search. Specifically,
//
// - `Foo` searches for `Foo` type in the current workspace
// - `foo#` searches for `foo` function in the current workspace
// - `Foo*` searches for `Foo` type among dependencies, including `stdlib`
// - `foo#*` searches for `foo` function among dependencies
//
// That is, `#` switches from "types" to all symbols, `*` switches from the current
// workspace to dependencies.
//
// Note that filtering does not currently work in VSCode due to the editor never
// sending the special symbols to the language server. Instead, you can configure
// the filtering via the `rust-analyzer.workspace.symbol.search.scope` and
// `rust-analyzer.workspace.symbol.search.kind` settings.
//
// |===
// | Editor  | Shortcut
//
// | VS Code | kbd:[Ctrl+T]
// |===
pub fn world_symbols(db: &RootDatabase, query: Query) -> Vec<FileSymbol> {
    let _p = profile::span("world_symbols").detail(|| query.query.clone());

    let indices: Vec<_> = if query.libs {
        db.library_roots()
            .par_iter()
            .map_with(Snap::new(db), |snap, &root| snap.library_symbols(root))
            .collect()
    } else {
        let mut module_ids = Vec::new();

        for &root in db.local_roots().iter() {
            let crates = db.source_root_crates(root);
            for &krate in crates.iter() {
                module_ids.extend(module_ids_for_crate(db, krate));
            }
        }

        module_ids
            .par_iter()
            .map_with(Snap::new(db), |snap, &module_id| snap.module_symbols(module_id))
            .collect()
    };

    query.search(&indices)
}

pub fn crate_symbols(db: &RootDatabase, krate: CrateId, query: Query) -> Vec<FileSymbol> {
    let _p = profile::span("crate_symbols").detail(|| format!("{:?}", query));

    let module_ids = module_ids_for_crate(db, krate);
    let indices: Vec<_> = module_ids
        .par_iter()
        .map_with(Snap::new(db), |snap, &module_id| snap.module_symbols(module_id))
        .collect();

    query.search(&indices)
}

fn module_ids_for_crate(db: &dyn DefDatabase, krate: CrateId) -> Vec<ModuleId> {
    let def_map = db.crate_def_map(krate);
    def_map.modules().map(|(id, _)| def_map.module_id(id)).collect()
}

pub fn index_resolve(db: &RootDatabase, name: &str) -> Vec<FileSymbol> {
    let mut query = Query::new(name.to_string());
    query.exact();
    query.limit(4);
    world_symbols(db, query)
}

#[derive(Default)]
pub struct SymbolIndex {
    symbols: Vec<FileSymbol>,
    map: fst::Map<Vec<u8>>,
}

impl fmt::Debug for SymbolIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SymbolIndex").field("n_symbols", &self.symbols.len()).finish()
    }
}

impl PartialEq for SymbolIndex {
    fn eq(&self, other: &SymbolIndex) -> bool {
        self.symbols == other.symbols
    }
}

impl Eq for SymbolIndex {}

impl Hash for SymbolIndex {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.symbols.hash(hasher)
    }
}

impl SymbolIndex {
    fn new(mut symbols: Vec<FileSymbol>) -> SymbolIndex {
        fn cmp(lhs: &FileSymbol, rhs: &FileSymbol) -> Ordering {
            let lhs_chars = lhs.name.chars().map(|c| c.to_ascii_lowercase());
            let rhs_chars = rhs.name.chars().map(|c| c.to_ascii_lowercase());
            lhs_chars.cmp(rhs_chars)
        }

        symbols.par_sort_by(cmp);

        let mut builder = fst::MapBuilder::memory();

        let mut last_batch_start = 0;

        for idx in 0..symbols.len() {
            if let Some(next_symbol) = symbols.get(idx + 1) {
                if cmp(&symbols[last_batch_start], next_symbol) == Ordering::Equal {
                    continue;
                }
            }

            let start = last_batch_start;
            let end = idx + 1;
            last_batch_start = end;

            let key = symbols[start].name.as_str().to_ascii_lowercase();
            let value = SymbolIndex::range_to_map_value(start, end);

            builder.insert(key, value).unwrap();
        }

        let map = fst::Map::new(builder.into_inner().unwrap()).unwrap();
        SymbolIndex { symbols, map }
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn memory_size(&self) -> usize {
        self.map.as_fst().size() + self.symbols.len() * mem::size_of::<FileSymbol>()
    }

    fn range_to_map_value(start: usize, end: usize) -> u64 {
        debug_assert![start <= (std::u32::MAX as usize)];
        debug_assert![end <= (std::u32::MAX as usize)];

        ((start as u64) << 32) | end as u64
    }

    fn map_value_to_range(value: u64) -> (usize, usize) {
        let end = value as u32 as usize;
        let start = (value >> 32) as usize;
        (start, end)
    }
}

impl Query {
    pub(crate) fn search(self, indices: &[Arc<SymbolIndex>]) -> Vec<FileSymbol> {
        let _p = profile::span("symbol_index::Query::search");
        let mut op = fst::map::OpBuilder::new();
        for file_symbols in indices.iter() {
            let automaton = fst::automaton::Subsequence::new(&self.lowercased);
            op = op.add(file_symbols.map.search(automaton))
        }
        let mut stream = op.union();
        let mut res = Vec::new();
        while let Some((_, indexed_values)) = stream.next() {
            for indexed_value in indexed_values {
                let symbol_index = &indices[indexed_value.index];
                let (start, end) = SymbolIndex::map_value_to_range(indexed_value.value);

                for symbol in &symbol_index.symbols[start..end] {
                    if self.only_types && !symbol.kind.is_type() {
                        continue;
                    }
                    if self.exact {
                        if symbol.name != self.query {
                            continue;
                        }
                    } else if self.case_sensitive {
                        if self.query.chars().any(|c| !symbol.name.contains(c)) {
                            continue;
                        }
                    }

                    res.push(symbol.clone());
                    if res.len() >= self.limit {
                        return res;
                    }
                }
            }
        }
        res
    }
}

/// The actual data that is stored in the index. It should be as compact as
/// possible.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileSymbol {
    pub name: SmolStr,
    pub loc: DeclarationLocation,
    pub kind: FileSymbolKind,
    pub container_name: Option<SmolStr>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeclarationLocation {
    /// The file id for both the `ptr` and `name_ptr`.
    pub hir_file_id: HirFileId,
    /// This points to the whole syntax node of the declaration.
    pub ptr: SyntaxNodePtr,
    /// This points to the [`syntax::ast::Name`] identifier of the declaration.
    pub name_ptr: SyntaxNodePtr,
}

impl DeclarationLocation {
    pub fn syntax(&self, semantics: &Semantics<'_, RootDatabase>) -> Option<SyntaxNode> {
        let root = semantics.parse_or_expand(self.hir_file_id)?;
        Some(self.ptr.to_node(&root))
    }

    pub fn original_range(&self, db: &dyn HirDatabase) -> Option<FileRange> {
        find_original_file_range(db, self.hir_file_id, &self.ptr)
    }

    pub fn original_name_range(&self, db: &dyn HirDatabase) -> Option<FileRange> {
        find_original_file_range(db, self.hir_file_id, &self.name_ptr)
    }
}

fn find_original_file_range(
    db: &dyn HirDatabase,
    file_id: HirFileId,
    ptr: &SyntaxNodePtr,
) -> Option<FileRange> {
    let root = db.parse_or_expand(file_id)?;
    let node = ptr.to_node(&root);
    let node = InFile::new(file_id, &node);

    Some(node.original_file_range(db.upcast()))
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub enum FileSymbolKind {
    Const,
    Enum,
    Function,
    Macro,
    Module,
    Static,
    Struct,
    Trait,
    TypeAlias,
    Union,
}

impl FileSymbolKind {
    fn is_type(self: FileSymbolKind) -> bool {
        matches!(
            self,
            FileSymbolKind::Struct
                | FileSymbolKind::Enum
                | FileSymbolKind::Trait
                | FileSymbolKind::TypeAlias
                | FileSymbolKind::Union
        )
    }
}

impl From<FileSymbolKind> for SymbolKind {
    fn from(it: FileSymbolKind) -> Self {
        match it {
            FileSymbolKind::Const => SymbolKind::Const,
            FileSymbolKind::Enum => SymbolKind::Enum,
            FileSymbolKind::Function => SymbolKind::Function,
            FileSymbolKind::Macro => SymbolKind::Macro,
            FileSymbolKind::Module => SymbolKind::Module,
            FileSymbolKind::Static => SymbolKind::Static,
            FileSymbolKind::Struct => SymbolKind::Struct,
            FileSymbolKind::Trait => SymbolKind::Trait,
            FileSymbolKind::TypeAlias => SymbolKind::TypeAlias,
            FileSymbolKind::Union => SymbolKind::Union,
        }
    }
}

/// Represents an outstanding module that the symbol collector must collect symbols from.
struct SymbolCollectorWork {
    module_id: ModuleId,
    parent: Option<DefWithBodyId>,
}

struct SymbolCollector<'a> {
    db: &'a dyn SymbolsDatabase,
    symbols: Vec<FileSymbol>,
    work: Vec<SymbolCollectorWork>,
    current_container_name: Option<SmolStr>,
}

/// Given a [`ModuleId`] and a [`SymbolsDatabase`], use the DefMap for the module's crate to collect all symbols that should be
/// indexed for the given module.
impl<'a> SymbolCollector<'a> {
    fn collect(db: &dyn SymbolsDatabase, module_id: ModuleId) -> Vec<FileSymbol> {
        let mut symbol_collector = SymbolCollector {
            db,
            symbols: Default::default(),
            current_container_name: None,
            // The initial work is the root module we're collecting, additional work will
            // be populated as we traverse the module's definitions.
            work: vec![SymbolCollectorWork { module_id, parent: None }],
        };

        while let Some(work) = symbol_collector.work.pop() {
            symbol_collector.do_work(work);
        }

        symbol_collector.symbols
    }

    fn do_work(&mut self, work: SymbolCollectorWork) {
        self.db.unwind_if_cancelled();

        let parent_name = work.parent.and_then(|id| self.def_with_body_id_name(id));
        self.with_container_name(parent_name, |s| s.collect_from_module(work.module_id));
    }

    fn collect_from_module(&mut self, module_id: ModuleId) {
        let def_map = module_id.def_map(self.db.upcast());
        let scope = &def_map[module_id.local_id].scope;

        for module_def_id in scope.declarations() {
            match module_def_id {
                ModuleDefId::ModuleId(id) => self.push_module(id),
                ModuleDefId::FunctionId(id) => {
                    self.push_decl_assoc(id, FileSymbolKind::Function);
                    self.collect_from_body(id);
                }
                ModuleDefId::AdtId(AdtId::StructId(id)) => {
                    self.push_decl(id, FileSymbolKind::Struct)
                }
                ModuleDefId::AdtId(AdtId::EnumId(id)) => self.push_decl(id, FileSymbolKind::Enum),
                ModuleDefId::AdtId(AdtId::UnionId(id)) => self.push_decl(id, FileSymbolKind::Union),
                ModuleDefId::ConstId(id) => {
                    self.push_decl_assoc(id, FileSymbolKind::Const);
                    self.collect_from_body(id);
                }
                ModuleDefId::StaticId(id) => {
                    self.push_decl_assoc(id, FileSymbolKind::Static);
                    self.collect_from_body(id);
                }
                ModuleDefId::TraitId(id) => {
                    self.push_decl(id, FileSymbolKind::Trait);
                    self.collect_from_trait(id);
                }
                ModuleDefId::TypeAliasId(id) => {
                    self.push_decl_assoc(id, FileSymbolKind::TypeAlias);
                }
                // Don't index these.
                ModuleDefId::BuiltinType(_) => {}
                ModuleDefId::EnumVariantId(_) => {}
            }
        }

        for impl_id in scope.impls() {
            self.collect_from_impl(impl_id);
        }

        for const_id in scope.unnamed_consts() {
            self.collect_from_body(const_id);
        }

        for macro_def_id in scope.macro_declarations() {
            self.push_decl_macro(macro_def_id.into());
        }
    }

    fn collect_from_body(&mut self, body_id: impl Into<DefWithBodyId>) {
        let body_id = body_id.into();
        let body = self.db.body(body_id);

        // Descend into the blocks and enqueue collection of all modules within.
        for (_, def_map) in body.blocks(self.db.upcast()) {
            for (id, _) in def_map.modules() {
                self.work.push(SymbolCollectorWork {
                    module_id: def_map.module_id(id),
                    parent: Some(body_id),
                });
            }
        }
    }

    fn collect_from_impl(&mut self, impl_id: ImplId) {
        let impl_data = self.db.impl_data(impl_id);
        for &assoc_item_id in &impl_data.items {
            self.push_assoc_item(assoc_item_id)
        }
    }

    fn collect_from_trait(&mut self, trait_id: TraitId) {
        let trait_data = self.db.trait_data(trait_id);
        self.with_container_name(trait_data.name.as_text(), |s| {
            for &(_, assoc_item_id) in &trait_data.items {
                s.push_assoc_item(assoc_item_id);
            }
        });
    }

    fn with_container_name(&mut self, container_name: Option<SmolStr>, f: impl FnOnce(&mut Self)) {
        if let Some(container_name) = container_name {
            let prev = self.current_container_name.replace(container_name);
            f(self);
            self.current_container_name = prev;
        } else {
            f(self);
        }
    }

    fn current_container_name(&self) -> Option<SmolStr> {
        self.current_container_name.clone()
    }

    fn def_with_body_id_name(&self, body_id: DefWithBodyId) -> Option<SmolStr> {
        match body_id {
            DefWithBodyId::FunctionId(id) => Some(
                id.lookup(self.db.upcast()).source(self.db.upcast()).value.name()?.text().into(),
            ),
            DefWithBodyId::StaticId(id) => Some(
                id.lookup(self.db.upcast()).source(self.db.upcast()).value.name()?.text().into(),
            ),
            DefWithBodyId::ConstId(id) => Some(
                id.lookup(self.db.upcast()).source(self.db.upcast()).value.name()?.text().into(),
            ),
        }
    }

    fn push_assoc_item(&mut self, assoc_item_id: AssocItemId) {
        match assoc_item_id {
            AssocItemId::FunctionId(id) => self.push_decl_assoc(id, FileSymbolKind::Function),
            AssocItemId::ConstId(id) => self.push_decl_assoc(id, FileSymbolKind::Const),
            AssocItemId::TypeAliasId(id) => self.push_decl_assoc(id, FileSymbolKind::TypeAlias),
        }
    }

    fn push_decl_assoc<L, T>(&mut self, id: L, kind: FileSymbolKind)
    where
        L: Lookup<Data = AssocItemLoc<T>>,
        T: ItemTreeNode,
        <T as ItemTreeNode>::Source: HasName,
    {
        fn container_name(db: &dyn HirDatabase, container: ItemContainerId) -> Option<SmolStr> {
            match container {
                ItemContainerId::ModuleId(module_id) => {
                    let module = Module::from(module_id);
                    module.name(db).and_then(|name| name.as_text())
                }
                ItemContainerId::TraitId(trait_id) => {
                    let trait_data = db.trait_data(trait_id);
                    trait_data.name.as_text()
                }
                ItemContainerId::ImplId(_) | ItemContainerId::ExternBlockId(_) => None,
            }
        }

        self.push_file_symbol(|s| {
            let loc = id.lookup(s.db.upcast());
            let source = loc.source(s.db.upcast());
            let name_node = source.value.name()?;
            let container_name =
                container_name(s.db.upcast(), loc.container).or_else(|| s.current_container_name());

            Some(FileSymbol {
                name: name_node.text().into(),
                kind,
                container_name,
                loc: DeclarationLocation {
                    hir_file_id: source.file_id,
                    ptr: SyntaxNodePtr::new(source.value.syntax()),
                    name_ptr: SyntaxNodePtr::new(name_node.syntax()),
                },
            })
        })
    }

    fn push_decl<L, T>(&mut self, id: L, kind: FileSymbolKind)
    where
        L: Lookup<Data = ItemLoc<T>>,
        T: ItemTreeNode,
        <T as ItemTreeNode>::Source: HasName,
    {
        self.push_file_symbol(|s| {
            let loc = id.lookup(s.db.upcast());
            let source = loc.source(s.db.upcast());
            let name_node = source.value.name()?;

            Some(FileSymbol {
                name: name_node.text().into(),
                kind,
                container_name: s.current_container_name(),
                loc: DeclarationLocation {
                    hir_file_id: source.file_id,
                    ptr: SyntaxNodePtr::new(source.value.syntax()),
                    name_ptr: SyntaxNodePtr::new(name_node.syntax()),
                },
            })
        })
    }

    fn push_module(&mut self, module_id: ModuleId) {
        self.push_file_symbol(|s| {
            let def_map = module_id.def_map(s.db.upcast());
            let module_data = &def_map[module_id.local_id];
            let declaration = module_data.origin.declaration()?;
            let module = declaration.to_node(s.db.upcast());
            let name_node = module.name()?;

            Some(FileSymbol {
                name: name_node.text().into(),
                kind: FileSymbolKind::Module,
                container_name: s.current_container_name(),
                loc: DeclarationLocation {
                    hir_file_id: declaration.file_id,
                    ptr: SyntaxNodePtr::new(module.syntax()),
                    name_ptr: SyntaxNodePtr::new(name_node.syntax()),
                },
            })
        })
    }

    fn push_decl_macro(&mut self, macro_def: MacroDef) {
        self.push_file_symbol(|s| {
            let name = macro_def.name(s.db.upcast())?.as_text()?;
            let source = macro_def.source(s.db.upcast())?;

            let (ptr, name_ptr) = match source.value {
                Either::Left(m) => {
                    (SyntaxNodePtr::new(m.syntax()), SyntaxNodePtr::new(m.name()?.syntax()))
                }
                Either::Right(f) => {
                    (SyntaxNodePtr::new(f.syntax()), SyntaxNodePtr::new(f.name()?.syntax()))
                }
            };

            Some(FileSymbol {
                name,
                kind: FileSymbolKind::Macro,
                container_name: s.current_container_name(),
                loc: DeclarationLocation { hir_file_id: source.file_id, name_ptr, ptr },
            })
        })
    }

    fn push_file_symbol(&mut self, f: impl FnOnce(&Self) -> Option<FileSymbol>) {
        if let Some(file_symbol) = f(self) {
            self.symbols.push(file_symbol);
        }
    }
}

#[cfg(test)]
mod tests {

    use base_db::fixture::WithFixture;
    use expect_test::expect_file;

    use super::*;

    #[test]
    fn test_symbol_index_collection() {
        let (db, _) = RootDatabase::with_many_files(
            r#"
//- /main.rs

macro_rules! macro_rules_macro {
    () => {}
};

macro_rules! define_struct {
    () => {
        struct StructFromMacro;
    }
};

define_struct!();

macro Macro { }

struct Struct;
enum Enum {
    A, B
}
union Union {}

impl Struct {
    fn impl_fn() {}
}

trait Trait {
    fn trait_fn(&self);
}

fn main() {
    struct StructInFn;
}

const CONST: u32 = 1;
static STATIC: &'static str = "2";
type Alias = Struct;

mod a_mod {
    struct StructInModA;
}

const _: () = {
    struct StructInUnnamedConst;

    ()
};

const CONST_WITH_INNER: () = {
    struct StructInNamedConst;

    ()
};

mod b_mod;

//- /b_mod.rs
struct StructInModB;
        "#,
        );

        let symbols: Vec<_> = module_ids_for_crate(db.upcast(), db.test_crate())
            .into_iter()
            .map(|module_id| {
                (module_id, SymbolCollector::collect(&db as &dyn SymbolsDatabase, module_id))
            })
            .collect();

        expect_file!["./test_data/test_symbol_index_collection.txt"].assert_debug_eq(&symbols);
    }
}
