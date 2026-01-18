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
    ops::ControlFlow,
};

use base_db::{CrateOrigin, LangCrateOrigin, LibraryRoots, LocalRoots, RootQueryDb, SourceRootId};
use fst::{Automaton, Streamer, raw::IndexedValue};
use hir::{
    Crate, Module,
    db::HirDatabase,
    import_map::{AssocSearchMode, SearchMode},
    symbols::{FileSymbol, SymbolCollector},
};
use itertools::Itertools;
use rayon::prelude::*;
use salsa::Update;

use crate::RootDatabase;

/// A query for searching symbols in the workspace or dependencies.
///
/// This struct configures how symbol search is performed, including the search text,
/// matching strategy, and filtering options. It is used by [`world_symbols`] to find
/// symbols across the codebase.
///
/// # Example
/// ```ignore
/// let mut query = Query::new("MyStruct".to_string());
/// query.only_types();  // Only search for type definitions
/// query.libs();        // Include library dependencies
/// query.exact();       // Use exact matching instead of fuzzy
/// ```
#[derive(Debug, Clone)]
pub struct Query {
    /// The item name to search for (last segment of the path, or full query if no path).
    /// When empty with a non-empty `path_filter`, returns all items in that module.
    query: String,
    /// Lowercase version of [`Self::query`], pre-computed for efficiency.
    /// Used to build FST automata for case-insensitive index lookups.
    lowercased: String,
    /// Path segments to filter by (all segments except the last).
    /// Empty if no `::` in the original query.
    path_filter: Vec<String>,
    /// If true, the first path segment must be a crate name (query started with `::`).
    anchor_to_crate: bool,
    /// The search strategy to use when matching symbols.
    /// - [`SearchMode::Exact`]: Symbol name must exactly match the query.
    /// - [`SearchMode::Fuzzy`]: Symbol name must contain all query characters in order (subsequence match).
    /// - [`SearchMode::Prefix`]: Symbol name must start with the query string.
    ///
    /// Defaults to [`SearchMode::Fuzzy`].
    mode: SearchMode,
    /// Controls filtering of trait-associated items (methods, constants, types).
    /// - [`AssocSearchMode::Include`]: Include both associated and non-associated items.
    /// - [`AssocSearchMode::Exclude`]: Exclude trait-associated items from results.
    /// - [`AssocSearchMode::AssocItemsOnly`]: Only return trait-associated items.
    ///
    /// Defaults to [`AssocSearchMode::Include`].
    assoc_mode: AssocSearchMode,
    /// Whether the final symbol name comparison should be case-sensitive.
    /// When `false`, matching is case-insensitive (e.g., "foo" matches "Foo").
    ///
    /// Defaults to `false`.
    case_sensitive: bool,
    /// When `true`, only return type definitions: structs, enums, unions,
    /// type aliases, built-in types, and traits. Functions, constants, statics,
    /// and modules are excluded.
    ///
    /// Defaults to `false`.
    only_types: bool,
    /// When `true`, search library dependency roots instead of local workspace crates.
    /// This enables finding symbols in external dependencies including the standard library.
    ///
    /// Defaults to `false` (search local workspace only).
    libs: bool,
    /// When `true`, exclude re-exported/imported symbols from results,
    /// showing only the original definitions.
    ///
    /// Defaults to `false`.
    exclude_imports: bool,
}

impl Query {
    pub fn new(query: String) -> Query {
        let (path_filter, item_query, anchor_to_crate) = Self::parse_path_query(&query);
        let lowercased = item_query.to_lowercase();
        Query {
            query: item_query,
            lowercased,
            path_filter,
            anchor_to_crate,
            only_types: false,
            libs: false,
            mode: SearchMode::Fuzzy,
            assoc_mode: AssocSearchMode::Include,
            case_sensitive: false,
            exclude_imports: false,
        }
    }

    /// Parse a query string that may contain path segments.
    ///
    /// Returns (path_filter, item_query, anchor_to_crate) where:
    /// - `path_filter`: Path segments to match (all but the last segment)
    /// - `item_query`: The item name to search for (last segment)
    /// - `anchor_to_crate`: Whether the first segment must be a crate name
    fn parse_path_query(query: &str) -> (Vec<String>, String, bool) {
        // Check for leading :: (absolute path / crate search)
        let (query, anchor_to_crate) = match query.strip_prefix("::") {
            Some(q) => (q, true),
            None => (query, false),
        };

        let Some((prefix, query)) = query.rsplit_once("::") else {
            return (vec![], query.to_owned(), anchor_to_crate);
        };

        let prefix: Vec<_> =
            prefix.split("::").filter(|s| !s.is_empty()).map(ToOwned::to_owned).collect();

        (prefix, query.to_owned(), anchor_to_crate)
    }

    /// Returns true if this query is searching for crates
    /// (i.e., the query was "::" alone or "::foo" for fuzzy crate search)
    fn is_crate_search(&self) -> bool {
        self.anchor_to_crate && self.path_filter.is_empty()
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

    pub fn exclude_imports(&mut self) {
        self.exclude_imports = true;
    }
}

/// The symbol indices of modules that make up a given crate.
pub fn crate_symbols(db: &dyn HirDatabase, krate: Crate) -> Box<[&SymbolIndex<'_>]> {
    let _p = tracing::info_span!("crate_symbols").entered();
    krate.modules(db).into_iter().map(|module| SymbolIndex::module_symbols(db, module)).collect()
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
// This also supports general Rust path syntax with the usual rules.
//
// Note that paths do not currently work in VSCode due to the editor never
// sending the special symbols to the language server. Some other editors might not support the # or
// * search either, instead, you can configure the filtering via the
// `rust-analyzer.workspace.symbol.search.scope` and `rust-analyzer.workspace.symbol.search.kind`
// settings. Symbols prefixed with `__` are hidden from the search results unless configured
// otherwise.
//
// | Editor  | Shortcut |
// |---------|-----------|
// | VS Code | <kbd>Ctrl+T</kbd>
pub fn world_symbols(db: &RootDatabase, mut query: Query) -> Vec<FileSymbol<'_>> {
    let _p = tracing::info_span!("world_symbols", query = ?query.query).entered();

    // Search for crates by name (handles "::" and "::foo" queries)
    let indices: Vec<_> = if query.is_crate_search() {
        query.only_types = false;
        vec![SymbolIndex::extern_prelude_symbols(db)]
        // If we have a path filter, resolve it to target modules
    } else if !query.path_filter.is_empty() {
        query.only_types = false;
        let target_modules = resolve_path_to_modules(
            db,
            &query.path_filter,
            query.anchor_to_crate,
            query.case_sensitive,
        );

        if target_modules.is_empty() {
            return vec![];
        }

        target_modules.iter().map(|&module| SymbolIndex::module_symbols(db, module)).collect()
    } else if query.libs {
        LibraryRoots::get(db)
            .roots(db)
            .par_iter()
            .for_each_with(db.clone(), |snap, &root| _ = SymbolIndex::library_symbols(snap, root));
        LibraryRoots::get(db)
            .roots(db)
            .iter()
            .map(|&root| SymbolIndex::library_symbols(db, root))
            .collect()
    } else {
        let mut crates = Vec::new();

        for &root in LocalRoots::get(db).roots(db).iter() {
            crates.extend(db.source_root_crates(root).iter().copied())
        }
        crates
            .par_iter()
            .for_each_with(db.clone(), |snap, &krate| _ = crate_symbols(snap, krate.into()));
        crates
            .into_iter()
            .flat_map(|krate| Vec::from(crate_symbols(db, krate.into())))
            .chain(std::iter::once(SymbolIndex::extern_prelude_symbols(db)))
            .collect()
    };

    let mut res = vec![];

    // Normal search: use FST to match item name
    query.search::<()>(db, &indices, |f| {
        res.push(f.clone());
        ControlFlow::Continue(())
    });

    res
}

/// Resolve a path filter to the target module(s) it points to.
/// Returns the modules whose symbol indices should be searched.
///
/// The path_filter contains segments like ["std", "vec"] for a query like "std::vec::Vec".
/// We resolve this by:
/// 1. Finding crates matching the first segment
/// 2. Walking down the module tree following subsequent segments
fn resolve_path_to_modules(
    db: &dyn HirDatabase,
    path_filter: &[String],
    anchor_to_crate: bool,
    case_sensitive: bool,
) -> Vec<Module> {
    let [first_segment, rest_segments @ ..] = path_filter else {
        return vec![];
    };

    // Helper for name comparison
    let names_match = |actual: &str, expected: &str| -> bool {
        if case_sensitive { actual == expected } else { actual.eq_ignore_ascii_case(expected) }
    };

    // Find crates matching the first segment
    let matching_crates: Vec<Crate> = Crate::all(db)
        .into_iter()
        .filter(|krate| {
            krate
                .display_name(db)
                .is_some_and(|name| names_match(name.crate_name().as_str(), first_segment))
        })
        .collect();

    // If anchor_to_crate is true, first segment MUST be a crate name
    // If anchor_to_crate is false, first segment could be a crate OR a module in local crates
    let mut candidate_modules: Vec<(Module, bool)> = vec![];

    // Add crate root modules for matching crates
    for krate in matching_crates {
        candidate_modules.push((krate.root_module(db), krate.origin(db).is_local()));
    }

    // If not anchored to crate, also search for modules matching first segment in local crates
    if !anchor_to_crate {
        for &root in LocalRoots::get(db).roots(db).iter() {
            for &krate in db.source_root_crates(root).iter() {
                let root_module = Crate::from(krate).root_module(db);
                for child in root_module.children(db) {
                    if let Some(name) = child.name(db)
                        && names_match(name.as_str(), first_segment)
                    {
                        candidate_modules.push((child, true));
                    }
                }
            }
        }
    }

    // Walk down the module tree for remaining path segments
    for segment in rest_segments {
        candidate_modules = candidate_modules
            .into_iter()
            .flat_map(|(module, local)| {
                module
                    .modules_in_scope(db, !local)
                    .into_iter()
                    .filter(|(name, _)| names_match(name.as_str(), segment))
                    .map(move |(_, module)| (module, local))
            })
            .unique()
            .collect();

        if candidate_modules.is_empty() {
            break;
        }
    }

    candidate_modules.into_iter().map(|(module, _)| module).collect()
}

#[derive(Default)]
pub struct SymbolIndex<'db> {
    symbols: Box<[FileSymbol<'db>]>,
    map: fst::Map<Vec<u8>>,
}

impl<'db> SymbolIndex<'db> {
    /// The symbol index for a given source root within library_roots.
    pub fn library_symbols(
        db: &'db dyn HirDatabase,
        source_root_id: SourceRootId,
    ) -> &'db SymbolIndex<'db> {
        // FIXME:
        #[salsa::interned]
        struct InternedSourceRootId {
            id: SourceRootId,
        }
        #[salsa::tracked(returns(ref))]
        fn library_symbols<'db>(
            db: &'db dyn HirDatabase,
            source_root_id: InternedSourceRootId<'db>,
        ) -> SymbolIndex<'db> {
            let _p = tracing::info_span!("library_symbols").entered();

            // We call this without attaching because this runs in parallel, so we need to attach here.
            hir::attach_db(db, || {
                let mut symbol_collector = SymbolCollector::new(db, true);

                db.source_root_crates(source_root_id.id(db))
                    .iter()
                    .flat_map(|&krate| Crate::from(krate).modules(db))
                    // we specifically avoid calling other SymbolsDatabase queries here, even though they do the same thing,
                    // as the index for a library is not going to really ever change, and we do not want to store
                    // the module or crate indices for those in salsa unless we need to.
                    .for_each(|module| symbol_collector.collect(module));

                SymbolIndex::new(symbol_collector.finish())
            })
        }
        library_symbols(db, InternedSourceRootId::new(db, source_root_id))
    }

    /// The symbol index for a given module. These modules should only be in source roots that
    /// are inside local_roots.
    pub fn module_symbols(db: &dyn HirDatabase, module: Module) -> &SymbolIndex<'_> {
        // FIXME:
        #[salsa::interned]
        struct InternedModuleId {
            id: hir::ModuleId,
        }

        #[salsa::tracked(returns(ref))]
        fn module_symbols<'db>(
            db: &'db dyn HirDatabase,
            module: InternedModuleId<'db>,
        ) -> SymbolIndex<'db> {
            let _p = tracing::info_span!("module_symbols").entered();

            // We call this without attaching because this runs in parallel, so we need to attach here.
            hir::attach_db(db, || {
                let module: Module = module.id(db).into();
                SymbolIndex::new(SymbolCollector::new_module(
                    db,
                    module,
                    !module.krate(db).origin(db).is_local(),
                ))
            })
        }

        module_symbols(db, InternedModuleId::new(db, hir::ModuleId::from(module)))
    }

    /// The symbol index for all extern prelude crates.
    pub fn extern_prelude_symbols(db: &dyn HirDatabase) -> &SymbolIndex<'_> {
        #[salsa::tracked(returns(ref))]
        fn extern_prelude_symbols<'db>(db: &'db dyn HirDatabase) -> SymbolIndex<'db> {
            let _p = tracing::info_span!("extern_prelude_symbols").entered();

            // We call this without attaching because this runs in parallel, so we need to attach here.
            hir::attach_db(db, || {
                let mut collector = SymbolCollector::new(db, false);

                for krate in Crate::all(db) {
                    if krate
                        .display_name(db)
                        .is_none_or(|name| name.canonical_name().as_str() == "build-script-build")
                    {
                        continue;
                    }
                    if let CrateOrigin::Lang(LangCrateOrigin::Dependency | LangCrateOrigin::Other) =
                        krate.origin(db)
                    {
                        // don't show dependencies of the sysroot
                        continue;
                    }
                    collector.push_crate_root(krate);
                }

                SymbolIndex::new(collector.finish())
            })
        }

        extern_prelude_symbols(db)
    }
}

impl fmt::Debug for SymbolIndex<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SymbolIndex").field("n_symbols", &self.symbols.len()).finish()
    }
}

impl PartialEq for SymbolIndex<'_> {
    fn eq(&self, other: &SymbolIndex<'_>) -> bool {
        self.symbols == other.symbols
    }
}

impl Eq for SymbolIndex<'_> {}

impl Hash for SymbolIndex<'_> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.symbols.hash(hasher)
    }
}

unsafe impl Update for SymbolIndex<'_> {
    unsafe fn maybe_update(old_pointer: *mut Self, new_value: Self) -> bool {
        let this = unsafe { &mut *old_pointer };
        if *this == new_value {
            false
        } else {
            *this = new_value;
            true
        }
    }
}

impl<'db> SymbolIndex<'db> {
    fn new(mut symbols: Box<[FileSymbol<'db>]>) -> SymbolIndex<'db> {
        fn cmp(lhs: &FileSymbol<'_>, rhs: &FileSymbol<'_>) -> Ordering {
            let lhs_chars = lhs.name.as_str().chars().map(|c| c.to_ascii_lowercase());
            let rhs_chars = rhs.name.as_str().chars().map(|c| c.to_ascii_lowercase());
            lhs_chars.cmp(rhs_chars)
        }

        symbols.par_sort_by(cmp);

        let mut builder = fst::MapBuilder::memory();

        let mut last_batch_start = 0;

        for idx in 0..symbols.len() {
            if let Some(next_symbol) = symbols.get(idx + 1)
                && cmp(&symbols[last_batch_start], next_symbol) == Ordering::Equal
            {
                continue;
            }

            let start = last_batch_start;
            let end = idx + 1;
            last_batch_start = end;

            let key = symbols[start].name.as_str().to_ascii_lowercase();
            let value = SymbolIndex::range_to_map_value(start, end);

            builder.insert(key, value).unwrap();
        }

        let map = builder
            .into_inner()
            .and_then(|mut buf| {
                fst::Map::new({
                    buf.shrink_to_fit();
                    buf
                })
            })
            .unwrap();
        SymbolIndex { symbols, map }
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn memory_size(&self) -> usize {
        self.map.as_fst().size() + self.symbols.len() * size_of::<FileSymbol<'_>>()
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
    /// Search symbols in the given indices.
    pub(crate) fn search<'db, T>(
        &self,
        db: &'db RootDatabase,
        indices: &[&'db SymbolIndex<'db>],
        cb: impl FnMut(&'db FileSymbol<'db>) -> ControlFlow<T>,
    ) -> Option<T> {
        let _p = tracing::info_span!("symbol_index::Query::search").entered();

        let mut op = fst::map::OpBuilder::new();
        match self.mode {
            SearchMode::Exact => {
                let automaton = fst::automaton::Str::new(&self.lowercased);

                for index in indices.iter() {
                    op = op.add(index.map.search(&automaton));
                }
                self.search_maps(db, indices, op.union(), cb)
            }
            SearchMode::Fuzzy => {
                let automaton = fst::automaton::Subsequence::new(&self.lowercased);

                for index in indices.iter() {
                    op = op.add(index.map.search(&automaton));
                }
                self.search_maps(db, indices, op.union(), cb)
            }
            SearchMode::Prefix => {
                let automaton = fst::automaton::Str::new(&self.lowercased).starts_with();

                for index in indices.iter() {
                    op = op.add(index.map.search(&automaton));
                }
                self.search_maps(db, indices, op.union(), cb)
            }
        }
    }

    fn search_maps<'db, T>(
        &self,
        db: &'db RootDatabase,
        indices: &[&'db SymbolIndex<'db>],
        mut stream: fst::map::Union<'_>,
        mut cb: impl FnMut(&'db FileSymbol<'db>) -> ControlFlow<T>,
    ) -> Option<T> {
        let ignore_underscore_prefixed = !self.query.starts_with("__");
        while let Some((_, indexed_values)) = stream.next() {
            for &IndexedValue { index, value } in indexed_values {
                let symbol_index = indices[index];
                let (start, end) = SymbolIndex::map_value_to_range(value);

                for symbol in &symbol_index.symbols[start..end] {
                    let non_type_for_type_only_query = self.only_types
                        && !(matches!(
                            symbol.def,
                            hir::ModuleDef::Adt(..)
                                | hir::ModuleDef::TypeAlias(..)
                                | hir::ModuleDef::BuiltinType(..)
                                | hir::ModuleDef::Trait(..)
                        ) || matches!(
                            symbol.def,
                            hir::ModuleDef::Module(module) if module.is_crate_root(db)
                        ));
                    if non_type_for_type_only_query || !self.matches_assoc_mode(symbol.is_assoc) {
                        continue;
                    }
                    // Hide symbols that start with `__` unless the query starts with `__`
                    let symbol_name = symbol.name.as_str();
                    if ignore_underscore_prefixed && symbol_name.starts_with("__") {
                        continue;
                    }
                    if self.exclude_imports && symbol.is_import {
                        continue;
                    }
                    if self.mode.check(&self.query, self.case_sensitive, symbol_name)
                        && let Some(b) = cb(symbol).break_value()
                    {
                        return Some(b);
                    }
                }
            }
        }
        None
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
    use rustc_hash::FxHashSet;
    use salsa::Setter;
    use test_fixture::{WORKSPACE, WithFixture};

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
pub(self) use super::Macro as SuperItemLikeMacro;
pub(self) use crate::b_mod::StructInModB as ThisStruct;
pub(self) use crate::Trait as IsThisJustATrait;
"#,
        );

        let symbols: Vec<_> = Crate::from(db.test_crate())
            .modules(&db)
            .into_iter()
            .map(|module_id| {
                let mut symbols = SymbolCollector::new_module(&db, module_id, false);
                symbols.sort_by_key(|it| it.name.as_str().to_owned());
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
                let mut symbols = SymbolCollector::new_module(&db, module_id, false);
                symbols.sort_by_key(|it| it.name.as_str().to_owned());
                (module_id, symbols)
            })
            .collect();

        expect_file!["./test_data/test_doc_alias.txt"].assert_debug_eq(&symbols);
    }

    #[test]
    fn test_exclude_imports() {
        let (mut db, _) = RootDatabase::with_many_files(
            r#"
//- /lib.rs
mod foo;
pub use foo::Foo;

//- /foo.rs
pub struct Foo;
"#,
        );

        let mut local_roots = FxHashSet::default();
        local_roots.insert(WORKSPACE);
        LocalRoots::get(&db).set_roots(&mut db).to(local_roots);

        let mut query = Query::new("Foo".to_owned());
        let mut symbols = world_symbols(&db, query.clone());
        symbols.sort_by_key(|x| x.is_import);
        expect_file!["./test_data/test_symbols_with_imports.txt"].assert_debug_eq(&symbols);

        query.exclude_imports();
        let symbols = world_symbols(&db, query);
        expect_file!["./test_data/test_symbols_exclude_imports.txt"].assert_debug_eq(&symbols);
    }

    #[test]
    fn test_parse_path_query() {
        // Plain query - no path
        let (path, item, anchor) = Query::parse_path_query("Item");
        assert_eq!(path, Vec::<String>::new());
        assert_eq!(item, "Item");
        assert!(!anchor);

        // Path with item
        let (path, item, anchor) = Query::parse_path_query("foo::Item");
        assert_eq!(path, vec!["foo"]);
        assert_eq!(item, "Item");
        assert!(!anchor);

        // Multi-segment path
        let (path, item, anchor) = Query::parse_path_query("foo::bar::Item");
        assert_eq!(path, vec!["foo", "bar"]);
        assert_eq!(item, "Item");
        assert!(!anchor);

        // Leading :: (anchor to crate)
        let (path, item, anchor) = Query::parse_path_query("::std::vec::Vec");
        assert_eq!(path, vec!["std", "vec"]);
        assert_eq!(item, "Vec");
        assert!(anchor);

        // Just "::" - return all crates
        let (path, item, anchor) = Query::parse_path_query("::");
        assert_eq!(path, Vec::<String>::new());
        assert_eq!(item, "");
        assert!(anchor);

        // "::foo" - fuzzy search crate names
        let (path, item, anchor) = Query::parse_path_query("::foo");
        assert_eq!(path, Vec::<String>::new());
        assert_eq!(item, "foo");
        assert!(anchor);

        // Trailing ::
        let (path, item, anchor) = Query::parse_path_query("foo::");
        assert_eq!(path, vec!["foo"]);
        assert_eq!(item, "");
        assert!(!anchor);

        // Full path with trailing ::
        let (path, item, anchor) = Query::parse_path_query("foo::bar::");
        assert_eq!(path, vec!["foo", "bar"]);
        assert_eq!(item, "");
        assert!(!anchor);

        // Absolute path with trailing ::
        let (path, item, anchor) = Query::parse_path_query("::std::vec::");
        assert_eq!(path, vec!["std", "vec"]);
        assert_eq!(item, "");
        assert!(anchor);

        // Empty segments should be filtered
        let (path, item, anchor) = Query::parse_path_query("foo::::bar");
        assert_eq!(path, vec!["foo"]);
        assert_eq!(item, "bar");
        assert!(!anchor);
    }

    #[test]
    fn test_path_search() {
        let (mut db, _) = RootDatabase::with_many_files(
            r#"
//- /lib.rs crate:main
mod inner;
pub struct RootStruct;

//- /inner.rs
pub struct InnerStruct;
pub mod nested {
    pub struct NestedStruct;
}
"#,
        );

        let mut local_roots = FxHashSet::default();
        local_roots.insert(WORKSPACE);
        LocalRoots::get(&db).set_roots(&mut db).to(local_roots);

        // Search for item in specific module
        let query = Query::new("inner::InnerStruct".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"InnerStruct"), "Expected InnerStruct in {:?}", names);

        // Search for item in nested module
        let query = Query::new("inner::nested::NestedStruct".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"NestedStruct"), "Expected NestedStruct in {:?}", names);

        // Search with crate prefix
        let query = Query::new("main::inner::InnerStruct".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"InnerStruct"), "Expected InnerStruct in {:?}", names);

        // Wrong path should return empty
        let query = Query::new("wrong::InnerStruct".to_owned());
        let symbols = world_symbols(&db, query);
        assert!(symbols.is_empty(), "Expected empty results for wrong path");
    }

    #[test]
    fn test_path_search_module() {
        let (mut db, _) = RootDatabase::with_many_files(
            r#"
//- /lib.rs crate:main
mod mymod;

//- /mymod.rs
pub struct MyStruct;
pub fn my_func() {}
pub const MY_CONST: u32 = 1;
"#,
        );

        let mut local_roots = FxHashSet::default();
        local_roots.insert(WORKSPACE);
        LocalRoots::get(&db).set_roots(&mut db).to(local_roots);

        // Browse all items in module
        let query = Query::new("main::mymod::".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();

        assert!(names.contains(&"MyStruct"), "Expected MyStruct in {:?}", names);
        assert!(names.contains(&"my_func"), "Expected my_func in {:?}", names);
        assert!(names.contains(&"MY_CONST"), "Expected MY_CONST in {:?}", names);
    }

    #[test]
    fn test_fuzzy_item_with_path() {
        let (mut db, _) = RootDatabase::with_many_files(
            r#"
//- /lib.rs crate:main
mod mymod;

//- /mymod.rs
pub struct MyLongStructName;
"#,
        );

        let mut local_roots = FxHashSet::default();
        local_roots.insert(WORKSPACE);
        LocalRoots::get(&db).set_roots(&mut db).to(local_roots);

        // Fuzzy match on item name with exact path
        let query = Query::new("main::mymod::MyLong".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(
            names.contains(&"MyLongStructName"),
            "Expected fuzzy match for MyLongStructName in {:?}",
            names
        );
    }

    #[test]
    fn test_case_insensitive_path() {
        let (mut db, _) = RootDatabase::with_many_files(
            r#"
//- /lib.rs crate:main
mod MyMod;

//- /MyMod.rs
pub struct MyStruct;
"#,
        );

        let mut local_roots = FxHashSet::default();
        local_roots.insert(WORKSPACE);
        LocalRoots::get(&db).set_roots(&mut db).to(local_roots);

        // Case insensitive path matching (default)
        let query = Query::new("main::mymod::MyStruct".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"MyStruct"), "Expected case-insensitive match in {:?}", names);
    }

    #[test]
    fn test_absolute_path_search() {
        let (mut db, _) = RootDatabase::with_many_files(
            r#"
//- /lib.rs crate:mycrate
mod inner;
pub struct CrateRoot;

//- /inner.rs
pub struct InnerItem;
"#,
        );

        let mut local_roots = FxHashSet::default();
        local_roots.insert(WORKSPACE);
        LocalRoots::get(&db).set_roots(&mut db).to(local_roots);

        // Absolute path with leading ::
        let query = Query::new("::mycrate::inner::InnerItem".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(
            names.contains(&"InnerItem"),
            "Expected InnerItem with absolute path in {:?}",
            names
        );

        // Absolute path should NOT match if crate name is wrong
        let query = Query::new("::wrongcrate::inner::InnerItem".to_owned());
        let symbols = world_symbols(&db, query);
        assert!(symbols.is_empty(), "Expected empty results for wrong crate name");
    }

    #[test]
    fn test_wrong_path_returns_empty() {
        let (mut db, _) = RootDatabase::with_many_files(
            r#"
//- /lib.rs crate:main
mod existing;

//- /existing.rs
pub struct MyStruct;
"#,
        );

        let mut local_roots = FxHashSet::default();
        local_roots.insert(WORKSPACE);
        LocalRoots::get(&db).set_roots(&mut db).to(local_roots);

        // Non-existent module path
        let query = Query::new("nonexistent::MyStruct".to_owned());
        let symbols = world_symbols(&db, query);
        assert!(symbols.is_empty(), "Expected empty results for non-existent path");

        // Correct item, wrong module
        let query = Query::new("wrongmod::MyStruct".to_owned());
        let symbols = world_symbols(&db, query);
        assert!(symbols.is_empty(), "Expected empty results for wrong module");
    }

    #[test]
    fn test_root_module_items() {
        let (mut db, _) = RootDatabase::with_many_files(
            r#"
//- /lib.rs crate:mylib
pub struct RootItem;
pub fn root_fn() {}
"#,
        );

        let mut local_roots = FxHashSet::default();
        local_roots.insert(WORKSPACE);
        LocalRoots::get(&db).set_roots(&mut db).to(local_roots);

        // Items at crate root - path is just the crate name
        let query = Query::new("mylib::RootItem".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"RootItem"), "Expected RootItem at crate root in {:?}", names);

        let query = Query::new("mylib::".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"RootItem"), "Expected RootItem {:?}", names);
        assert!(names.contains(&"root_fn"), "Expected root_fn {:?}", names);
    }

    #[test]
    fn test_crate_search_all() {
        // Test that sole "::" returns all crates
        let (mut db, _) = RootDatabase::with_many_files(
            r#"
//- /lib.rs crate:alpha
pub struct AlphaStruct;

//- /beta.rs crate:beta
pub struct BetaStruct;

//- /gamma.rs crate:gamma
pub struct GammaStruct;
"#,
        );

        let mut local_roots = FxHashSet::default();
        local_roots.insert(WORKSPACE);
        LocalRoots::get(&db).set_roots(&mut db).to(local_roots);

        // Sole "::" should return all crates (as module symbols)
        let query = Query::new("::".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();

        assert!(names.contains(&"alpha"), "Expected alpha crate in {:?}", names);
        assert!(names.contains(&"beta"), "Expected beta crate in {:?}", names);
        assert!(names.contains(&"gamma"), "Expected gamma crate in {:?}", names);
        assert_eq!(symbols.len(), 3, "Expected exactly 3 crates, got {:?}", names);
    }

    #[test]
    fn test_crate_search_fuzzy() {
        // Test that "::foo" fuzzy-matches crate names
        let (mut db, _) = RootDatabase::with_many_files(
            r#"
//- /lib.rs crate:my_awesome_lib
pub struct AwesomeStruct;

//- /other.rs crate:another_lib
pub struct OtherStruct;

//- /foo.rs crate:foobar
pub struct FooStruct;
"#,
        );

        let mut local_roots = FxHashSet::default();
        local_roots.insert(WORKSPACE);
        LocalRoots::get(&db).set_roots(&mut db).to(local_roots);

        // "::foo" should fuzzy-match crate names containing "foo"
        let query = Query::new("::foo".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();

        assert!(names.contains(&"foobar"), "Expected foobar crate in {:?}", names);
        assert_eq!(symbols.len(), 1, "Expected only foobar crate, got {:?}", names);

        // "::awesome" should match my_awesome_lib
        let query = Query::new("::awesome".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();

        assert!(names.contains(&"my_awesome_lib"), "Expected my_awesome_lib crate in {:?}", names);
        assert_eq!(symbols.len(), 1, "Expected only my_awesome_lib crate, got {:?}", names);

        // "::lib" should match multiple crates
        let query = Query::new("::lib".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();

        assert!(names.contains(&"my_awesome_lib"), "Expected my_awesome_lib in {:?}", names);
        assert!(names.contains(&"another_lib"), "Expected another_lib in {:?}", names);
        assert_eq!(symbols.len(), 2, "Expected 2 crates matching 'lib', got {:?}", names);

        // "::nonexistent" should return empty
        let query = Query::new("::nonexistent".to_owned());
        let symbols = world_symbols(&db, query);
        assert!(symbols.is_empty(), "Expected empty results for non-matching crate pattern");
    }

    #[test]
    fn test_path_search_with_use_reexport() {
        // Test that module resolution works for `use` items (re-exports), not just `mod` items
        let (mut db, _) = RootDatabase::with_many_files(
            r#"
//- /lib.rs crate:main
mod inner;
pub use inner::nested;

//- /inner.rs
pub mod nested {
    pub struct NestedStruct;
    pub fn nested_fn() {}
}
"#,
        );

        let mut local_roots = FxHashSet::default();
        local_roots.insert(WORKSPACE);
        LocalRoots::get(&db).set_roots(&mut db).to(local_roots);

        // Search via the re-exported path (main::nested::NestedStruct)
        // This should work because `nested` is in scope via `pub use inner::nested`
        let query = Query::new("main::nested::NestedStruct".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(
            names.contains(&"NestedStruct"),
            "Expected NestedStruct via re-exported path in {:?}",
            names
        );

        // Also verify the original path still works
        let query = Query::new("main::inner::nested::NestedStruct".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(
            names.contains(&"NestedStruct"),
            "Expected NestedStruct via original path in {:?}",
            names
        );

        // Browse the re-exported module
        let query = Query::new("main::nested::".to_owned());
        let symbols = world_symbols(&db, query);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(
            names.contains(&"NestedStruct"),
            "Expected NestedStruct when browsing re-exported module in {:?}",
            names
        );
        assert!(
            names.contains(&"nested_fn"),
            "Expected nested_fn when browsing re-exported module in {:?}",
            names
        );
    }
}
