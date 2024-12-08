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
};

use base_db::{
    ra_salsa::{self, ParallelDatabase},
    SourceRootDatabase, SourceRootId, Upcast,
};
use fst::{raw::IndexedValue, Automaton, Streamer};
use hir::{
    db::HirDatabase,
    import_map::{AssocSearchMode, SearchMode},
    symbols::{FileSymbol, SymbolCollector},
    Crate, Module,
};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use triomphe::Arc;

use crate::RootDatabase;

#[derive(Debug, Clone)]
pub struct Query {
    query: String,
    lowercased: String,
    mode: SearchMode,
    assoc_mode: AssocSearchMode,
    case_sensitive: bool,
    only_types: bool,
    libs: bool,
}

impl Query {
    pub fn new(query: String) -> Query {
        let lowercased = query.to_lowercase();
        Query {
            query,
            lowercased,
            only_types: false,
            libs: false,
            mode: SearchMode::Fuzzy,
            assoc_mode: AssocSearchMode::Include,
            case_sensitive: false,
        }
    }

    pub fn only_types(&mut self) {
        self.only_types = true;
    }

    pub fn libs(&mut self) {
        self.libs = true;
    }

    pub fn fuzzy(&mut self) {
        self.mode = SearchMode::Fuzzy;
    }

    pub fn exact(&mut self) {
        self.mode = SearchMode::Exact;
    }

    pub fn prefix(&mut self) {
        self.mode = SearchMode::Prefix;
    }

    /// Specifies whether we want to include associated items in the result.
    pub fn assoc_search_mode(&mut self, assoc_mode: AssocSearchMode) {
        self.assoc_mode = assoc_mode;
    }

    pub fn case_sensitive(&mut self) {
        self.case_sensitive = true;
    }
}

#[ra_salsa::query_group(SymbolsDatabaseStorage)]
pub trait SymbolsDatabase: HirDatabase + SourceRootDatabase + Upcast<dyn HirDatabase> {
    /// The symbol index for a given module. These modules should only be in source roots that
    /// are inside local_roots.
    fn module_symbols(&self, module: Module) -> Arc<SymbolIndex>;

    /// The symbol index for a given source root within library_roots.
    fn library_symbols(&self, source_root_id: SourceRootId) -> Arc<SymbolIndex>;

    #[ra_salsa::transparent]
    /// The symbol indices of modules that make up a given crate.
    fn crate_symbols(&self, krate: Crate) -> Box<[Arc<SymbolIndex>]>;

    /// The set of "local" (that is, from the current workspace) roots.
    /// Files in local roots are assumed to change frequently.
    #[ra_salsa::input]
    fn local_roots(&self) -> Arc<FxHashSet<SourceRootId>>;

    /// The set of roots for crates.io libraries.
    /// Files in libraries are assumed to never change.
    #[ra_salsa::input]
    fn library_roots(&self) -> Arc<FxHashSet<SourceRootId>>;
}

fn library_symbols(db: &dyn SymbolsDatabase, source_root_id: SourceRootId) -> Arc<SymbolIndex> {
    let _p = tracing::info_span!("library_symbols").entered();

    let mut symbol_collector = SymbolCollector::new(db.upcast());

    db.source_root_crates(source_root_id)
        .iter()
        .flat_map(|&krate| Crate::from(krate).modules(db.upcast()))
        // we specifically avoid calling other SymbolsDatabase queries here, even though they do the same thing,
        // as the index for a library is not going to really ever change, and we do not want to store each
        // the module or crate indices for those in salsa unless we need to.
        .for_each(|module| symbol_collector.collect(module));

    let mut symbols = symbol_collector.finish();
    symbols.shrink_to_fit();
    Arc::new(SymbolIndex::new(symbols))
}

fn module_symbols(db: &dyn SymbolsDatabase, module: Module) -> Arc<SymbolIndex> {
    let _p = tracing::info_span!("module_symbols").entered();

    let symbols = SymbolCollector::collect_module(db.upcast(), module);
    Arc::new(SymbolIndex::new(symbols))
}

pub fn crate_symbols(db: &dyn SymbolsDatabase, krate: Crate) -> Box<[Arc<SymbolIndex>]> {
    let _p = tracing::info_span!("crate_symbols").entered();
    krate.modules(db.upcast()).into_iter().map(|module| db.module_symbols(module)).collect()
}

/// Need to wrap Snapshot to provide `Clone` impl for `map_with`
struct Snap<DB>(DB);
impl<DB: ParallelDatabase> Snap<ra_salsa::Snapshot<DB>> {
    fn new(db: &DB) -> Self {
        Self(db.snapshot())
    }
}
impl<DB: ParallelDatabase> Clone for Snap<ra_salsa::Snapshot<DB>> {
    fn clone(&self) -> Snap<ra_salsa::Snapshot<DB>> {
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
// `rust-analyzer.workspace.symbol.search.kind` settings. Symbols prefixed
// with `__` are hidden from the search results unless configured otherwise.
//
// |===
// | Editor  | Shortcut
//
// | VS Code | kbd:[Ctrl+T]
// |===
pub fn world_symbols(db: &RootDatabase, query: Query) -> Vec<FileSymbol> {
    let _p = tracing::info_span!("world_symbols", query = ?query.query).entered();

    let indices: Vec<_> = if query.libs {
        db.library_roots()
            .par_iter()
            .map_with(Snap::new(db), |snap, &root| snap.library_symbols(root))
            .collect()
    } else {
        let mut crates = Vec::new();

        for &root in db.local_roots().iter() {
            crates.extend(db.source_root_crates(root).iter().copied())
        }
        let indices: Vec<_> = crates
            .into_par_iter()
            .map_with(Snap::new(db), |snap, krate| snap.crate_symbols(krate.into()))
            .collect();
        indices.iter().flat_map(|indices| indices.iter().cloned()).collect()
    };

    let mut res = vec![];
    query.search(&indices, |f| res.push(f.clone()));
    res
}

#[derive(Default)]
pub struct SymbolIndex {
    symbols: Vec<FileSymbol>,
    map: fst::Map<Vec<u8>>,
}

impl fmt::Debug for SymbolIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

        // FIXME: fst::Map should ideally have a way to shrink the backing buffer without the unwrap dance
        let map = fst::Map::new({
            let mut buf = builder.into_inner().unwrap();
            buf.shrink_to_fit();
            buf
        })
        .unwrap();
        SymbolIndex { symbols, map }
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn memory_size(&self) -> usize {
        self.map.as_fst().size() + self.symbols.len() * mem::size_of::<FileSymbol>()
    }

    fn range_to_map_value(start: usize, end: usize) -> u64 {
        debug_assert![start <= (u32::MAX as usize)];
        debug_assert![end <= (u32::MAX as usize)];

        ((start as u64) << 32) | end as u64
    }

    fn map_value_to_range(value: u64) -> (usize, usize) {
        let end = value as u32 as usize;
        let start = (value >> 32) as usize;
        (start, end)
    }
}

impl Query {
    pub(crate) fn search<'sym>(
        self,
        indices: &'sym [Arc<SymbolIndex>],
        cb: impl FnMut(&'sym FileSymbol),
    ) {
        let _p = tracing::info_span!("symbol_index::Query::search").entered();
        let mut op = fst::map::OpBuilder::new();
        match self.mode {
            SearchMode::Exact => {
                let automaton = fst::automaton::Str::new(&self.lowercased);

                for index in indices.iter() {
                    op = op.add(index.map.search(&automaton));
                }
                self.search_maps(indices, op.union(), cb)
            }
            SearchMode::Fuzzy => {
                let automaton = fst::automaton::Subsequence::new(&self.lowercased);

                for index in indices.iter() {
                    op = op.add(index.map.search(&automaton));
                }
                self.search_maps(indices, op.union(), cb)
            }
            SearchMode::Prefix => {
                let automaton = fst::automaton::Str::new(&self.lowercased).starts_with();

                for index in indices.iter() {
                    op = op.add(index.map.search(&automaton));
                }
                self.search_maps(indices, op.union(), cb)
            }
        }
    }

    fn search_maps<'sym>(
        &self,
        indices: &'sym [Arc<SymbolIndex>],
        mut stream: fst::map::Union<'_>,
        mut cb: impl FnMut(&'sym FileSymbol),
    ) {
        let ignore_underscore_prefixed = !self.query.starts_with("__");
        while let Some((_, indexed_values)) = stream.next() {
            for &IndexedValue { index, value } in indexed_values {
                let symbol_index = &indices[index];
                let (start, end) = SymbolIndex::map_value_to_range(value);

                for symbol in &symbol_index.symbols[start..end] {
                    let non_type_for_type_only_query = self.only_types
                        && !matches!(
                            symbol.def,
                            hir::ModuleDef::Adt(..)
                                | hir::ModuleDef::TypeAlias(..)
                                | hir::ModuleDef::BuiltinType(..)
                                | hir::ModuleDef::TraitAlias(..)
                                | hir::ModuleDef::Trait(..)
                        );
                    if non_type_for_type_only_query || !self.matches_assoc_mode(symbol.is_assoc) {
                        continue;
                    }
                    // Hide symbols that start with `__` unless the query starts with `__`
                    if ignore_underscore_prefixed && symbol.name.starts_with("__") {
                        continue;
                    }
                    if self.mode.check(&self.query, self.case_sensitive, &symbol.name) {
                        cb(symbol);
                    }
                }
            }
        }
    }

    fn matches_assoc_mode(&self, is_trait_assoc_item: bool) -> bool {
        !matches!(
            (is_trait_assoc_item, self.assoc_mode),
            (true, AssocSearchMode::Exclude) | (false, AssocSearchMode::AssocItemsOnly)
        )
    }
}

#[cfg(test)]
mod tests {

    use expect_test::expect_file;
    use test_fixture::WithFixture;

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

struct StructT<T>;

impl <T> StructT<T> {
    fn generic_impl_fn() {}
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


use define_struct as really_define_struct;
use Macro as ItemLikeMacro;
use Macro as Trait; // overlay namespaces
//- /b_mod.rs
struct StructInModB;
use super::Macro as SuperItemLikeMacro;
use crate::b_mod::StructInModB as ThisStruct;
use crate::Trait as IsThisJustATrait;
"#,
        );

        let symbols: Vec<_> = Crate::from(db.test_crate())
            .modules(&db)
            .into_iter()
            .map(|module_id| {
                let mut symbols = SymbolCollector::collect_module(&db, module_id);
                symbols.sort_by_key(|it| it.name.clone());
                (module_id, symbols)
            })
            .collect();

        expect_file!["./test_data/test_symbol_index_collection.txt"].assert_debug_eq(&symbols);
    }

    #[test]
    fn test_doc_alias() {
        let (db, _) = RootDatabase::with_single_file(
            r#"
#[doc(alias="s1")]
#[doc(alias="s2")]
#[doc(alias("mul1","mul2"))]
struct Struct;

#[doc(alias="s1")]
struct Duplicate;
        "#,
        );

        let symbols: Vec<_> = Crate::from(db.test_crate())
            .modules(&db)
            .into_iter()
            .map(|module_id| {
                let mut symbols = SymbolCollector::collect_module(&db, module_id);
                symbols.sort_by_key(|it| it.name.clone());
                (module_id, symbols)
            })
            .collect();

        expect_file!["./test_data/test_doc_alias.txt"].assert_debug_eq(&symbols);
    }
}
