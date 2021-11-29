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
    CrateId, FileId, FileRange, SourceDatabaseExt, SourceRootId, Upcast,
};
use fst::{self, Streamer};
use hir::{
    db::DefDatabase, AdtId, AssocContainerId, AssocItemId, AssocItemLoc, DefHasSource,
    DefWithBodyId, HirFileId, InFile, ItemLoc, ItemScope, ItemTreeNode, Lookup, ModuleData,
    ModuleDefId, ModuleId, Semantics,
};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use syntax::{
    ast::{self, HasName},
    AstNode, Parse, SmolStr, SourceFile, SyntaxNode, SyntaxNodePtr,
};

use crate::RootDatabase;

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
pub trait SymbolsDatabase: hir::db::HirDatabase + SourceDatabaseExt {
    fn module_symbols(&self, module_id: ModuleId) -> Arc<SymbolIndex>;
    fn library_symbols(&self) -> Arc<FxHashMap<SourceRootId, SymbolIndex>>;
    /// The set of "local" (that is, from the current workspace) roots.
    /// Files in local roots are assumed to change frequently.
    #[salsa::input]
    fn local_roots(&self) -> Arc<FxHashSet<SourceRootId>>;
    /// The set of roots for crates.io libraries.
    /// Files in libraries are assumed to never change.
    #[salsa::input]
    fn library_roots(&self) -> Arc<FxHashSet<SourceRootId>>;
}

fn library_symbols(db: &dyn SymbolsDatabase) -> Arc<FxHashMap<SourceRootId, SymbolIndex>> {
    let _p = profile::span("library_symbols");

    let roots = db.library_roots();
    let res = roots
        .iter()
        .map(|&root_id| {
            let root = db.source_root(root_id);
            let files = root
                .iter()
                .map(|it| (it, SourceDatabaseExt::file_text(db, it)))
                .collect::<Vec<_>>();
            let symbol_index = SymbolIndex::for_files(
                files.into_par_iter().map(|(file, text)| (file, SourceFile::parse(&text))),
            );
            (root_id, symbol_index)
        })
        .collect();
    Arc::new(res)
}

fn module_symbols(db: &dyn SymbolsDatabase, module_id: ModuleId) -> Arc<SymbolIndex> {
    db.unwind_if_cancelled();

    let def_map = module_id.def_map(db.upcast());
    let module_data = &def_map[module_id.local_id];

    let symbols = module_data_to_file_symbols(db.upcast(), module_data);

    Arc::new(SymbolIndex::new(symbols))
}

/// Need to wrap Snapshot to provide `Clone` impl for `map_with`
struct Snap<DB>(DB);
impl<DB: ParallelDatabase> Clone for Snap<salsa::Snapshot<DB>> {
    fn clone(&self) -> Snap<salsa::Snapshot<DB>> {
        Snap(self.0.snapshot())
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

    let tmp1;
    let tmp2;
    let buf: Vec<&SymbolIndex> = if query.libs {
        tmp1 = db.library_symbols();
        tmp1.values().collect()
    } else {
        let mut module_ids = Vec::new();

        for &root in db.local_roots().iter() {
            let crates = db.source_root_crates(root);
            for &krate in crates.iter() {
                module_ids.extend(module_ids_for_crate(db, krate));
            }
        }

        let snap = Snap(db.snapshot());
        tmp2 = module_ids
            .par_iter()
            .map_with(snap, |snap, &module_id| snap.0.module_symbols(module_id))
            .collect::<Vec<_>>();
        tmp2.iter().map(|it| &**it).collect()
    };
    query.search(&buf)
}

pub fn crate_symbols(db: &RootDatabase, krate: CrateId, query: Query) -> Vec<FileSymbol> {
    let _p = profile::span("crate_symbols").detail(|| format!("{:?}", query));

    let module_ids = module_ids_for_crate(db, krate);
    let snap = Snap(db.snapshot());
    let buf: Vec<_> = module_ids
        .par_iter()
        .map_with(snap, |snap, &module_id| snap.0.module_symbols(module_id))
        .collect();

    for i in &buf {
        dbg!(&i.symbols);
    }

    let buf = buf.iter().map(|it| &**it).collect::<Vec<_>>();
    query.search(&buf)
}

fn module_ids_for_crate(db: &RootDatabase, krate: CrateId) -> Vec<ModuleId> {
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

    pub(crate) fn for_files(
        files: impl ParallelIterator<Item = (FileId, Parse<ast::SourceFile>)>,
    ) -> SymbolIndex {
        let symbols = files
            .flat_map(|(file_id, file)| source_file_to_file_symbols(&file.tree(), file_id))
            .collect::<Vec<_>>();
        SymbolIndex::new(symbols)
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
    pub(crate) fn search(self, indices: &[&SymbolIndex]) -> Vec<FileSymbol> {
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

    pub fn original_range(&self, semantics: &Semantics<'_, RootDatabase>) -> Option<FileRange> {
        find_original_file_range(semantics, self.hir_file_id, &self.ptr)
    }

    pub fn original_name_range(
        &self,
        semantics: &Semantics<'_, RootDatabase>,
    ) -> Option<FileRange> {
        find_original_file_range(semantics, self.hir_file_id, &self.name_ptr)
    }
}

fn find_original_file_range(
    semantics: &Semantics<'_, RootDatabase>,
    file_id: HirFileId,
    ptr: &SyntaxNodePtr,
) -> Option<FileRange> {
    let root = semantics.parse_or_expand(file_id)?;
    let node = ptr.to_node(&root);
    let node = InFile::new(file_id, &node);

    Some(node.original_file_range(semantics.db.upcast()))
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

fn source_file_to_file_symbols(_source_file: &SourceFile, _file_id: FileId) -> Vec<FileSymbol> {
    // todo: delete this.
    vec![]
}

fn module_data_to_file_symbols(db: &dyn DefDatabase, module_data: &ModuleData) -> Vec<FileSymbol> {
    let mut symbols = Vec::new();
    collect_symbols_from_item_scope(db, &mut symbols, &module_data.scope);
    // todo: collect macros from scope.macros().
    symbols
}

fn collect_symbols_from_item_scope(
    db: &dyn DefDatabase,
    symbols: &mut Vec<FileSymbol>,
    scope: &ItemScope,
) {
    fn container_name(db: &dyn DefDatabase, container: AssocContainerId) -> Option<SmolStr> {
        match container {
            AssocContainerId::ModuleId(module_id) => {
                let def_map = module_id.def_map(db);
                let module_data = &def_map[module_id.local_id];
                module_data
                    .origin
                    .declaration()
                    .and_then(|s| s.to_node(db.upcast()).name().map(|n| n.text().into()))
            }
            AssocContainerId::TraitId(trait_id) => {
                let loc = trait_id.lookup(db);
                let source = loc.source(db);
                source.value.name().map(|n| n.text().into())
            }
            AssocContainerId::ImplId(_) => None,
        }
    }

    fn decl_assoc<L, T>(db: &dyn DefDatabase, id: L, kind: FileSymbolKind) -> Option<FileSymbol>
    where
        L: Lookup<Data = AssocItemLoc<T>>,
        T: ItemTreeNode,
        <T as ItemTreeNode>::Source: HasName,
    {
        let loc = id.lookup(db);
        let source = loc.source(db);
        let name_node = source.value.name()?;
        let container_name = container_name(db, loc.container);

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
    }

    fn decl<L, T>(db: &dyn DefDatabase, id: L, kind: FileSymbolKind) -> Option<FileSymbol>
    where
        L: Lookup<Data = ItemLoc<T>>,
        T: ItemTreeNode,
        <T as ItemTreeNode>::Source: HasName,
    {
        let loc = id.lookup(db);
        let source = loc.source(db);
        let name_node = source.value.name()?;

        Some(FileSymbol {
            name: name_node.text().into(),
            kind,
            container_name: None,
            loc: DeclarationLocation {
                hir_file_id: source.file_id,
                ptr: SyntaxNodePtr::new(source.value.syntax()),
                name_ptr: SyntaxNodePtr::new(name_node.syntax()),
            },
        })
    }

    fn decl_module(db: &dyn DefDatabase, module_id: ModuleId) -> Option<FileSymbol> {
        let def_map = module_id.def_map(db);
        let module_data = &def_map[module_id.local_id];
        let declaration = module_data.origin.declaration()?;
        let module = declaration.to_node(db.upcast());
        let name_node = module.name()?;

        Some(FileSymbol {
            name: name_node.text().into(),
            kind: FileSymbolKind::Module,
            container_name: None,
            loc: DeclarationLocation {
                hir_file_id: declaration.file_id,
                ptr: SyntaxNodePtr::new(module.syntax()),
                name_ptr: SyntaxNodePtr::new(name_node.syntax()),
            },
        })
    }

    let collect_symbols_from_scope =
        |scope: &ItemScope,
         symbols: &mut Vec<FileSymbol>,
         bodies_to_traverse: &mut Vec<(Option<SmolStr>, DefWithBodyId)>,
         container_name: &Option<SmolStr>| {
            let mut trait_ids = Vec::new();

            let scope_declaration_symbols = scope
                .declarations()
                .filter_map(|module_def_id| match module_def_id {
                    ModuleDefId::ModuleId(module_id) => decl_module(db, module_id),
                    ModuleDefId::FunctionId(function_id) => {
                        let symbol = decl_assoc(db, function_id, FileSymbolKind::Function);
                        bodies_to_traverse.push((
                            symbol.as_ref().and_then(|x| Some(x.name.clone())),
                            function_id.into(),
                        ));
                        symbol
                    }
                    ModuleDefId::AdtId(AdtId::StructId(struct_id)) => {
                        decl(db, struct_id, FileSymbolKind::Struct)
                    }
                    ModuleDefId::AdtId(AdtId::EnumId(enum_id)) => {
                        decl(db, enum_id, FileSymbolKind::Enum)
                    }
                    ModuleDefId::AdtId(AdtId::UnionId(union_id)) => {
                        decl(db, union_id, FileSymbolKind::Union)
                    }
                    ModuleDefId::ConstId(const_id) => {
                        let symbol = decl_assoc(db, const_id, FileSymbolKind::Const);
                        bodies_to_traverse.push((
                            symbol.as_ref().and_then(|x| Some(x.name.clone())),
                            const_id.into(),
                        ));
                        symbol
                    }
                    ModuleDefId::StaticId(static_id) => {
                        let symbol = decl(db, static_id, FileSymbolKind::Static);
                        bodies_to_traverse.push((
                            symbol.as_ref().and_then(|x| Some(x.name.clone())),
                            static_id.into(),
                        ));
                        symbol
                    }
                    ModuleDefId::TraitId(trait_id) => {
                        trait_ids.push(trait_id);
                        decl(db, trait_id, FileSymbolKind::Trait)
                    }
                    ModuleDefId::TypeAliasId(alias_id) => {
                        decl_assoc(db, alias_id, FileSymbolKind::TypeAlias)
                    }
                    ModuleDefId::BuiltinType(_) => None,
                    ModuleDefId::EnumVariantId(_) => None,
                })
                .map(|mut s| {
                    // If a container name was not provided in the symbol, but within the scope of our traversal,
                    // we'll update the container name here.
                    if let Some(container_name) = &container_name {
                        s.container_name.get_or_insert_with(|| container_name.clone());
                    }

                    s
                });

            symbols.extend(scope_declaration_symbols);

            // todo: we need to merge in container name to these too.
            // also clean this up generally tooooo.
            let scope_impl_symbols = scope
                .impls()
                .map(|impl_id| db.impl_data(impl_id))
                .flat_map(|d| d.items.clone()) // xx: clean up this clone??
                .filter_map(|assoc_item_id| match assoc_item_id {
                    AssocItemId::FunctionId(function_id) => {
                        decl_assoc(db, function_id, FileSymbolKind::Function)
                    }
                    AssocItemId::ConstId(const_id) => {
                        decl_assoc(db, const_id, FileSymbolKind::Const)
                    }
                    AssocItemId::TypeAliasId(type_alias_id) => {
                        decl_assoc(db, type_alias_id, FileSymbolKind::TypeAlias)
                    }
                });

            symbols.extend(scope_impl_symbols);

            // todo: we need to merge in container name to these too.
            // also clean this up generally tooooo.
            let scope_trait_symbols = trait_ids
                .into_iter()
                .map(|trait_id| db.trait_data(trait_id))
                .flat_map(|d| d.items.clone())
                .filter_map(|(_, assoc_item_id)| match assoc_item_id {
                    AssocItemId::FunctionId(function_id) => {
                        decl_assoc(db, function_id, FileSymbolKind::Function)
                    }
                    AssocItemId::ConstId(const_id) => {
                        decl_assoc(db, const_id, FileSymbolKind::Const)
                    }
                    AssocItemId::TypeAliasId(type_alias_id) => {
                        decl_assoc(db, type_alias_id, FileSymbolKind::TypeAlias)
                    }
                });

            symbols.extend(scope_trait_symbols);

            for const_id in scope.unnamed_consts() {
                // since unnamed consts don't really have a name, we'll inherit parent scope's symbol name.
                bodies_to_traverse.push((container_name.clone(), const_id.into()));
            }
        };

    let mut bodies_to_traverse = Vec::new();
    collect_symbols_from_scope(scope, symbols, &mut bodies_to_traverse, &None);

    while let Some((container_name, body)) = bodies_to_traverse.pop() {
        let body = db.body(body);

        for (_, block_def_map) in body.blocks(db) {
            for (_, module_data) in block_def_map.modules() {
                collect_symbols_from_scope(
                    &module_data.scope,
                    symbols,
                    &mut bodies_to_traverse,
                    &container_name,
                );
            }
        }
    }
}
