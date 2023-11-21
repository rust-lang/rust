//! A map of all publicly exported items in a crate.

use std::{fmt, hash::BuildHasherDefault};

use base_db::CrateId;
use fst::{self, raw::IndexedValue, Streamer};
use hir_expand::name::Name;
use indexmap::IndexMap;
use itertools::Itertools;
use rustc_hash::{FxHashSet, FxHasher};
use smallvec::SmallVec;
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    item_scope::{ImportOrExternCrate, ItemInNs},
    nameres::DefMap,
    visibility::Visibility,
    AssocItemId, ModuleDefId, ModuleId, TraitId,
};

type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

/// Item import details stored in the `ImportMap`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ImportInfo {
    /// A name that can be used to import the item, relative to the crate's root.
    pub name: Name,
    /// The module containing this item.
    pub container: ModuleId,
    /// Whether this item is annotated with `#[doc(hidden)]`.
    pub is_doc_hidden: bool,
    /// Whether this item is annotated with `#[unstable(..)]`.
    pub is_unstable: bool,
}

type ImportMapIndex = FxIndexMap<ItemInNs, (SmallVec<[ImportInfo; 1]>, IsTraitAssocItem)>;

/// A map from publicly exported items to its name.
///
/// Reexports of items are taken into account, ie. if something is exported under multiple
/// names, the one with the shortest import path will be used.
#[derive(Default)]
pub struct ImportMap {
    map: ImportMapIndex,
    /// List of keys stored in `map`, sorted lexicographically by their `ModPath`. Indexed by the
    /// values returned by running `fst`.
    ///
    /// Since a name can refer to multiple items due to namespacing, we store all items with the
    /// same name right after each other. This allows us to find all items after the FST gives us
    /// the index of the first one.
    importables: Vec<ItemInNs>,
    fst: fst::Map<Vec<u8>>,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum IsTraitAssocItem {
    Yes,
    No,
}

impl ImportMap {
    pub(crate) fn import_map_query(db: &dyn DefDatabase, krate: CrateId) -> Arc<Self> {
        let _p = profile::span("import_map_query");

        let map = collect_import_map(db, krate);

        let mut importables: Vec<_> = map
            .iter()
            // We've only collected items, whose name cannot be tuple field.
            .flat_map(|(&item, (info, _))| {
                info.iter()
                    .map(move |info| (item, info.name.as_str().unwrap().to_ascii_lowercase()))
            })
            .collect();
        importables.sort_by(|(_, lhs_name), (_, rhs_name)| lhs_name.cmp(rhs_name));
        importables.dedup();

        // Build the FST, taking care not to insert duplicate values.
        let mut builder = fst::MapBuilder::memory();
        let iter =
            importables.iter().enumerate().dedup_by(|(_, (_, lhs)), (_, (_, rhs))| lhs == rhs);
        for (start_idx, (_, name)) in iter {
            let _ = builder.insert(name, start_idx as u64);
        }

        Arc::new(ImportMap {
            map,
            fst: builder.into_map(),
            importables: importables.into_iter().map(|(item, _)| item).collect(),
        })
    }

    pub fn import_info_for(&self, item: ItemInNs) -> Option<&[ImportInfo]> {
        self.map.get(&item).map(|(info, _)| &**info)
    }
}

fn collect_import_map(db: &dyn DefDatabase, krate: CrateId) -> ImportMapIndex {
    let _p = profile::span("collect_import_map");

    let def_map = db.crate_def_map(krate);
    let mut map = FxIndexMap::default();

    // We look only into modules that are public(ly reexported), starting with the crate root.
    let root = def_map.module_id(DefMap::ROOT);
    let mut worklist = vec![root];
    let mut visited = FxHashSet::default();

    while let Some(module) = worklist.pop() {
        if !visited.insert(module) {
            continue;
        }
        let ext_def_map;
        let mod_data = if module.krate == krate {
            &def_map[module.local_id]
        } else {
            // The crate might reexport a module defined in another crate.
            ext_def_map = module.def_map(db);
            &ext_def_map[module.local_id]
        };

        let visible_items = mod_data.scope.entries().filter_map(|(name, per_ns)| {
            let per_ns = per_ns.filter_visibility(|vis| vis == Visibility::Public);
            if per_ns.is_none() { None } else { Some((name, per_ns)) }
        });

        for (name, per_ns) in visible_items {
            for (item, import) in per_ns.iter_items() {
                let attr_id = if let Some(import) = import {
                    match import {
                        ImportOrExternCrate::ExternCrate(id) => Some(id.into()),
                        ImportOrExternCrate::Import(id) => Some(id.import.into()),
                    }
                } else {
                    match item {
                        ItemInNs::Types(id) | ItemInNs::Values(id) => id.try_into().ok(),
                        ItemInNs::Macros(id) => Some(id.into()),
                    }
                };
                let (is_doc_hidden, is_unstable) = attr_id.map_or((false, false), |attr_id| {
                    let attrs = db.attrs(attr_id);
                    (attrs.has_doc_hidden(), attrs.is_unstable())
                });

                let import_info = ImportInfo {
                    name: name.clone(),
                    container: module,
                    is_doc_hidden,
                    is_unstable,
                };

                if let Some(ModuleDefId::TraitId(tr)) = item.as_module_def_id() {
                    collect_trait_assoc_items(
                        db,
                        &mut map,
                        tr,
                        matches!(item, ItemInNs::Types(_)),
                        &import_info,
                    );
                }

                let (infos, _) =
                    map.entry(item).or_insert_with(|| (SmallVec::new(), IsTraitAssocItem::No));
                infos.reserve_exact(1);
                infos.push(import_info);

                // If we've just added a module, descend into it.
                if let Some(ModuleDefId::ModuleId(mod_id)) = item.as_module_def_id() {
                    worklist.push(mod_id);
                }
            }
        }
    }
    map.shrink_to_fit();
    map
}

fn collect_trait_assoc_items(
    db: &dyn DefDatabase,
    map: &mut ImportMapIndex,
    tr: TraitId,
    is_type_in_ns: bool,
    trait_import_info: &ImportInfo,
) {
    let _p = profile::span("collect_trait_assoc_items");
    for &(ref assoc_item_name, item) in &db.trait_data(tr).items {
        let module_def_id = match item {
            AssocItemId::FunctionId(f) => ModuleDefId::from(f),
            AssocItemId::ConstId(c) => ModuleDefId::from(c),
            // cannot use associated type aliases directly: need a `<Struct as Trait>::TypeAlias`
            // qualifier, ergo no need to store it for imports in import_map
            AssocItemId::TypeAliasId(_) => {
                cov_mark::hit!(type_aliases_ignored);
                continue;
            }
        };
        let assoc_item = if is_type_in_ns {
            ItemInNs::Types(module_def_id)
        } else {
            ItemInNs::Values(module_def_id)
        };

        let attrs = &db.attrs(item.into());
        let assoc_item_info = ImportInfo {
            container: trait_import_info.container,
            name: assoc_item_name.clone(),
            is_doc_hidden: attrs.has_doc_hidden(),
            is_unstable: attrs.is_unstable(),
        };

        let (infos, _) =
            map.entry(assoc_item).or_insert_with(|| (SmallVec::new(), IsTraitAssocItem::Yes));
        infos.reserve_exact(1);
        infos.push(assoc_item_info);
    }
}

impl PartialEq for ImportMap {
    fn eq(&self, other: &Self) -> bool {
        // `fst` and `importables` are built from `map`, so we don't need to compare them.
        self.map == other.map
    }
}

impl Eq for ImportMap {}

impl fmt::Debug for ImportMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut importable_names: Vec<_> = self
            .map
            .iter()
            .map(|(item, (infos, _))| {
                let l = infos.len();
                match item {
                    ItemInNs::Types(it) => format!("- {it:?} (t) [{l}]",),
                    ItemInNs::Values(it) => format!("- {it:?} (v) [{l}]",),
                    ItemInNs::Macros(it) => format!("- {it:?} (m) [{l}]",),
                }
            })
            .collect();

        importable_names.sort();
        f.write_str(&importable_names.join("\n"))
    }
}

/// A way to match import map contents against the search query.
#[derive(Copy, Clone, Debug)]
enum SearchMode {
    /// Import map entry should strictly match the query string.
    Exact,
    /// Import map entry should contain all letters from the query string,
    /// in the same order, but not necessary adjacent.
    Fuzzy,
    /// Import map entry should match the query string by prefix.
    Prefix,
}

/// Three possible ways to search for the name in associated and/or other items.
#[derive(Debug, Clone, Copy)]
pub enum AssocSearchMode {
    /// Search for the name in both associated and other items.
    Include,
    /// Search for the name in other items only.
    Exclude,
    /// Search for the name in the associated items only.
    AssocItemsOnly,
}

#[derive(Debug)]
pub struct Query {
    query: String,
    lowercased: String,
    search_mode: SearchMode,
    assoc_mode: AssocSearchMode,
    case_sensitive: bool,
    limit: usize,
}

impl Query {
    pub fn new(query: String) -> Self {
        let lowercased = query.to_lowercase();
        Self {
            query,
            lowercased,
            search_mode: SearchMode::Exact,
            assoc_mode: AssocSearchMode::Include,
            case_sensitive: false,
            limit: usize::MAX,
        }
    }

    /// Fuzzy finds items instead of exact matching.
    pub fn fuzzy(self) -> Self {
        Self { search_mode: SearchMode::Fuzzy, ..self }
    }

    pub fn prefix(self) -> Self {
        Self { search_mode: SearchMode::Prefix, ..self }
    }

    pub fn exact(self) -> Self {
        Self { search_mode: SearchMode::Exact, ..self }
    }

    /// Specifies whether we want to include associated items in the result.
    pub fn assoc_search_mode(self, assoc_mode: AssocSearchMode) -> Self {
        Self { assoc_mode, ..self }
    }

    /// Limits the returned number of items to `limit`.
    pub fn limit(self, limit: usize) -> Self {
        Self { limit, ..self }
    }

    /// Respect casing of the query string when matching.
    pub fn case_sensitive(self) -> Self {
        Self { case_sensitive: true, ..self }
    }

    fn matches_assoc_mode(&self, is_trait_assoc_item: IsTraitAssocItem) -> bool {
        match (is_trait_assoc_item, self.assoc_mode) {
            (IsTraitAssocItem::Yes, AssocSearchMode::Exclude)
            | (IsTraitAssocItem::No, AssocSearchMode::AssocItemsOnly) => false,
            _ => true,
        }
    }

    /// Checks whether the import map entry matches the query.
    fn import_matches(
        &self,
        db: &dyn DefDatabase,
        import: &ImportInfo,
        enforce_lowercase: bool,
    ) -> bool {
        let _p = profile::span("import_map::Query::import_matches");

        // FIXME: Can we get rid of the alloc here?
        let mut input = import.name.display(db.upcast()).to_string();
        let case_insensitive = enforce_lowercase || !self.case_sensitive;
        if case_insensitive {
            input.make_ascii_lowercase();
        }

        let query_string = if case_insensitive { &self.lowercased } else { &self.query };

        match self.search_mode {
            SearchMode::Exact => input == *query_string,
            SearchMode::Prefix => input.starts_with(query_string),
            SearchMode::Fuzzy => {
                let mut input_chars = input.chars();
                for query_char in query_string.chars() {
                    if input_chars.find(|&it| it == query_char).is_none() {
                        return false;
                    }
                }
                true
            }
        }
    }
}

/// Searches dependencies of `krate` for an importable name matching `query`.
///
/// This returns a list of items that could be imported from dependencies of `krate`.
pub fn search_dependencies(
    db: &dyn DefDatabase,
    krate: CrateId,
    ref query: Query,
) -> FxHashSet<ItemInNs> {
    let _p = profile::span("search_dependencies").detail(|| format!("{query:?}"));

    let graph = db.crate_graph();
    let import_maps: Vec<_> =
        graph[krate].dependencies.iter().map(|dep| db.import_map(dep.crate_id)).collect();

    let automaton = fst::automaton::Subsequence::new(&query.lowercased);

    let mut op = fst::map::OpBuilder::new();
    for map in &import_maps {
        op = op.add(map.fst.search(&automaton));
    }

    let mut stream = op.union();

    let mut res = FxHashSet::default();
    let mut common_importable_data_scratch = vec![];
    while let Some((_, indexed_values)) = stream.next() {
        for &IndexedValue { index, value } in indexed_values {
            let import_map = &import_maps[index];
            let importables @ [importable, ..] = &import_map.importables[value as usize..] else {
                continue;
            };

            let &(ref importable_data, is_trait_assoc_item) = &import_map.map[importable];
            if !query.matches_assoc_mode(is_trait_assoc_item) {
                continue;
            }

            common_importable_data_scratch.extend(
                importable_data
                    .iter()
                    .filter(|&info| query.import_matches(db, info, true))
                    // Name shared by the importable items in this group.
                    .map(|info| info.name.to_smol_str()),
            );
            if common_importable_data_scratch.is_empty() {
                continue;
            }
            common_importable_data_scratch.sort();
            common_importable_data_scratch.dedup();

            let iter =
                common_importable_data_scratch.drain(..).flat_map(|common_importable_name| {
                    // Add the items from this name group. Those are all subsequent items in
                    // `importables` whose name match `common_importable_name`.
                    importables
                        .iter()
                        .copied()
                        .take_while(move |item| {
                            let &(ref import_infos, assoc_mode) = &import_map.map[item];
                            query.matches_assoc_mode(assoc_mode)
                                && import_infos.iter().any(|info| {
                                    info.name
                                        .to_smol_str()
                                        .eq_ignore_ascii_case(&common_importable_name)
                                })
                        })
                        .filter(move |item| {
                            !query.case_sensitive || {
                                // we've already checked the common importables name case-insensitively
                                let &(ref import_infos, assoc_mode) = &import_map.map[item];
                                query.matches_assoc_mode(assoc_mode)
                                    && import_infos
                                        .iter()
                                        .any(|info| query.import_matches(db, info, false))
                            }
                        })
                });
            res.extend(iter);

            if res.len() >= query.limit {
                return res;
            }
        }
    }

    res
}

#[cfg(test)]
mod tests {
    use base_db::{fixture::WithFixture, SourceDatabase, Upcast};
    use expect_test::{expect, Expect};

    use crate::{db::DefDatabase, test_db::TestDB, ItemContainerId, Lookup};

    use super::*;

    impl ImportMap {
        fn fmt_for_test(&self, db: &dyn DefDatabase) -> String {
            let mut importable_paths: Vec<_> = self
                .map
                .iter()
                .flat_map(|(item, (info, _))| info.iter().map(move |info| (item, info)))
                .map(|(item, info)| {
                    let path = render_path(db, info);
                    let ns = match item {
                        ItemInNs::Types(_) => "t",
                        ItemInNs::Values(_) => "v",
                        ItemInNs::Macros(_) => "m",
                    };
                    format!("- {path} ({ns})")
                })
                .collect();

            importable_paths.sort();
            importable_paths.join("\n")
        }
    }

    fn check_search(ra_fixture: &str, crate_name: &str, query: Query, expect: Expect) {
        let db = TestDB::with_files(ra_fixture);
        let crate_graph = db.crate_graph();
        let krate = crate_graph
            .iter()
            .find(|&krate| {
                crate_graph[krate]
                    .display_name
                    .as_ref()
                    .is_some_and(|it| &**it.crate_name() == crate_name)
            })
            .expect("could not find crate");

        let actual = search_dependencies(db.upcast(), krate, query)
            .into_iter()
            .filter_map(|dependency| {
                let dependency_krate = dependency.krate(db.upcast())?;
                let dependency_imports = db.import_map(dependency_krate);

                let (path, mark) = match assoc_item_path(&db, &dependency_imports, dependency) {
                    Some(assoc_item_path) => (assoc_item_path, "a"),
                    None => (
                        render_path(&db, &dependency_imports.import_info_for(dependency)?[0]),
                        match dependency {
                            ItemInNs::Types(ModuleDefId::FunctionId(_))
                            | ItemInNs::Values(ModuleDefId::FunctionId(_)) => "f",
                            ItemInNs::Types(_) => "t",
                            ItemInNs::Values(_) => "v",
                            ItemInNs::Macros(_) => "m",
                        },
                    ),
                };

                Some(format!(
                    "{}::{} ({})\n",
                    crate_graph[dependency_krate].display_name.as_ref()?,
                    path,
                    mark
                ))
            })
            // HashSet iteration order isn't defined - it's different on
            // x86_64 and i686 at the very least
            .sorted()
            .collect::<String>();
        expect.assert_eq(&actual)
    }

    fn assoc_item_path(
        db: &dyn DefDatabase,
        dependency_imports: &ImportMap,
        dependency: ItemInNs,
    ) -> Option<String> {
        let (dependency_assoc_item_id, container) = match dependency.as_module_def_id()? {
            ModuleDefId::FunctionId(id) => (AssocItemId::from(id), id.lookup(db).container),
            ModuleDefId::ConstId(id) => (AssocItemId::from(id), id.lookup(db).container),
            ModuleDefId::TypeAliasId(id) => (AssocItemId::from(id), id.lookup(db).container),
            _ => return None,
        };

        let ItemContainerId::TraitId(trait_id) = container else {
            return None;
        };

        let trait_info = dependency_imports.import_info_for(ItemInNs::Types(trait_id.into()))?;

        let trait_data = db.trait_data(trait_id);
        let (assoc_item_name, _) = trait_data
            .items
            .iter()
            .find(|(_, assoc_item_id)| &dependency_assoc_item_id == assoc_item_id)?;
        // FIXME: This should check all import infos, not just the first
        Some(format!(
            "{}::{}",
            render_path(db, &trait_info[0]),
            assoc_item_name.display(db.upcast())
        ))
    }

    fn check(ra_fixture: &str, expect: Expect) {
        let db = TestDB::with_files(ra_fixture);
        let crate_graph = db.crate_graph();

        let actual = crate_graph
            .iter()
            .filter_map(|krate| {
                let cdata = &crate_graph[krate];
                let name = cdata.display_name.as_ref()?;

                let map = db.import_map(krate);

                Some(format!("{name}:\n{}\n", map.fmt_for_test(db.upcast())))
            })
            .sorted()
            .collect::<String>();

        expect.assert_eq(&actual)
    }

    fn render_path(db: &dyn DefDatabase, info: &ImportInfo) -> String {
        let mut module = info.container;
        let mut segments = vec![&info.name];

        let def_map = module.def_map(db);
        assert!(def_map.block_id().is_none(), "block local items should not be in `ImportMap`");

        while let Some(parent) = module.containing_module(db) {
            let parent_data = &def_map[parent.local_id];
            let (name, _) =
                parent_data.children.iter().find(|(_, id)| **id == module.local_id).unwrap();
            segments.push(name);
            module = parent;
        }

        segments.iter().rev().map(|it| it.display(db.upcast())).join("::")
    }

    #[test]
    fn smoke() {
        check(
            r"
            //- /main.rs crate:main deps:lib

            mod private {
                pub use lib::Pub;
                pub struct InPrivateModule;
            }

            pub mod publ1 {
                use lib::Pub;
            }

            pub mod real_pub {
                pub use lib::Pub;
            }
            pub mod real_pu2 { // same path length as above
                pub use lib::Pub;
            }

            //- /lib.rs crate:lib
            pub struct Pub {}
            pub struct Pub2; // t + v
            struct Priv;
        ",
            expect![[r#"
                lib:
                - Pub (t)
                - Pub2 (t)
                - Pub2 (v)
                main:
                - publ1 (t)
                - real_pu2 (t)
                - real_pu2::Pub (t)
                - real_pub (t)
                - real_pub::Pub (t)
            "#]],
        );
    }

    #[test]
    fn prefers_shortest_path() {
        check(
            r"
            //- /main.rs crate:main

            pub mod sub {
                pub mod subsub {
                    pub struct Def {}
                }

                pub use super::sub::subsub::Def;
            }
        ",
            expect![[r#"
                main:
                - sub (t)
                - sub::Def (t)
                - sub::subsub (t)
                - sub::subsub::Def (t)
            "#]],
        );
    }

    #[test]
    fn type_reexport_cross_crate() {
        // Reexports need to be visible from a crate, even if the original crate exports the item
        // at a shorter path.
        check(
            r"
            //- /main.rs crate:main deps:lib
            pub mod m {
                pub use lib::S;
            }
            //- /lib.rs crate:lib
            pub struct S;
        ",
            expect![[r#"
                lib:
                - S (t)
                - S (v)
                main:
                - m (t)
                - m::S (t)
                - m::S (v)
            "#]],
        );
    }

    #[test]
    fn macro_reexport() {
        check(
            r"
            //- /main.rs crate:main deps:lib
            pub mod m {
                pub use lib::pub_macro;
            }
            //- /lib.rs crate:lib
            #[macro_export]
            macro_rules! pub_macro {
                () => {};
            }
        ",
            expect![[r#"
                lib:
                - pub_macro (m)
                main:
                - m (t)
                - m::pub_macro (m)
            "#]],
        );
    }

    #[test]
    fn module_reexport() {
        // Reexporting modules from a dependency adds all contents to the import map.
        // XXX: The rendered paths are relative to the defining crate.
        check(
            r"
            //- /main.rs crate:main deps:lib
            pub use lib::module as reexported_module;
            //- /lib.rs crate:lib
            pub mod module {
                pub struct S;
            }
        ",
            expect![[r#"
                lib:
                - module (t)
                - module::S (t)
                - module::S (v)
                main:
                - module::S (t)
                - module::S (v)
                - reexported_module (t)
            "#]],
        );
    }

    #[test]
    fn cyclic_module_reexport() {
        // A cyclic reexport does not hang.
        check(
            r"
            //- /lib.rs crate:lib
            pub mod module {
                pub struct S;
                pub use super::sub::*;
            }

            pub mod sub {
                pub use super::module;
            }
        ",
            expect![[r#"
                lib:
                - module (t)
                - module::S (t)
                - module::S (v)
                - module::module (t)
                - sub (t)
                - sub::module (t)
            "#]],
        );
    }

    #[test]
    fn private_macro() {
        check(
            r"
            //- /lib.rs crate:lib
            macro_rules! private_macro {
                () => {};
            }
        ",
            expect![[r#"
                lib:

            "#]],
        );
    }

    #[test]
    fn namespacing() {
        check(
            r"
            //- /lib.rs crate:lib
            pub struct Thing;     // t + v
            #[macro_export]
            macro_rules! Thing {  // m
                () => {};
            }
        ",
            expect![[r#"
                lib:
                - Thing (m)
                - Thing (t)
                - Thing (v)
            "#]],
        );

        check(
            r"
            //- /lib.rs crate:lib
            pub mod Thing {}      // t
            #[macro_export]
            macro_rules! Thing {  // m
                () => {};
            }
        ",
            expect![[r#"
                lib:
                - Thing (m)
                - Thing (t)
            "#]],
        );
    }

    #[test]
    fn fuzzy_import_trait_and_assoc_items() {
        cov_mark::check!(type_aliases_ignored);
        let ra_fixture = r#"
        //- /main.rs crate:main deps:dep
        //- /dep.rs crate:dep
        pub mod fmt {
            pub trait Display {
                type FmtTypeAlias;
                const FMT_CONST: bool;

                fn format_function();
                fn format_method(&self);
            }
        }
    "#;

        check_search(
            ra_fixture,
            "main",
            Query::new("fmt".to_string()).fuzzy(),
            expect![[r#"
                dep::fmt (t)
                dep::fmt::Display::FMT_CONST (a)
                dep::fmt::Display::format_function (a)
                dep::fmt::Display::format_method (a)
            "#]],
        );
    }

    #[test]
    fn assoc_items_filtering() {
        let ra_fixture = r#"
        //- /main.rs crate:main deps:dep
        //- /dep.rs crate:dep
        pub mod fmt {
            pub trait Display {
                type FmtTypeAlias;
                const FMT_CONST: bool;

                fn format_function();
                fn format_method(&self);
            }
        }
    "#;

        check_search(
            ra_fixture,
            "main",
            Query::new("fmt".to_string())
                .fuzzy()
                .assoc_search_mode(AssocSearchMode::AssocItemsOnly),
            expect![[r#"
                dep::fmt::Display::FMT_CONST (a)
                dep::fmt::Display::format_function (a)
                dep::fmt::Display::format_method (a)
            "#]],
        );

        check_search(
            ra_fixture,
            "main",
            Query::new("fmt".to_string()).fuzzy().assoc_search_mode(AssocSearchMode::Exclude),
            expect![[r#"
                dep::fmt (t)
            "#]],
        );
    }

    #[test]
    fn search_mode() {
        let ra_fixture = r#"
            //- /main.rs crate:main deps:dep
            //- /dep.rs crate:dep deps:tdep
            use tdep::fmt as fmt_dep;
            pub mod fmt {
                pub trait Display {
                    fn fmt();
                }
            }
            #[macro_export]
            macro_rules! Fmt {
                () => {};
            }
            pub struct Fmt;

            pub fn format() {}
            pub fn no() {}

            //- /tdep.rs crate:tdep
            pub mod fmt {
                pub struct NotImportableFromMain;
            }
        "#;

        check_search(
            ra_fixture,
            "main",
            Query::new("fmt".to_string()).fuzzy(),
            expect![[r#"
                dep::Fmt (m)
                dep::Fmt (t)
                dep::Fmt (v)
                dep::fmt (t)
                dep::fmt::Display::fmt (a)
                dep::format (f)
            "#]],
        );

        check_search(
            ra_fixture,
            "main",
            Query::new("fmt".to_string()),
            expect![[r#"
                dep::Fmt (m)
                dep::Fmt (t)
                dep::Fmt (v)
                dep::fmt (t)
                dep::fmt::Display::fmt (a)
            "#]],
        );
    }

    #[test]
    fn name_only() {
        let ra_fixture = r#"
            //- /main.rs crate:main deps:dep
            //- /dep.rs crate:dep deps:tdep
            use tdep::fmt as fmt_dep;
            pub mod fmt {
                pub trait Display {
                    fn fmt();
                }
            }
            #[macro_export]
            macro_rules! Fmt {
                () => {};
            }
            pub struct Fmt;

            pub fn format() {}
            pub fn no() {}

            //- /tdep.rs crate:tdep
            pub mod fmt {
                pub struct NotImportableFromMain;
            }
        "#;

        check_search(
            ra_fixture,
            "main",
            Query::new("fmt".to_string()),
            expect![[r#"
                dep::Fmt (m)
                dep::Fmt (t)
                dep::Fmt (v)
                dep::fmt (t)
                dep::fmt::Display::fmt (a)
            "#]],
        );

        check_search(
            ra_fixture,
            "main",
            Query::new("fmt".to_string()),
            expect![[r#"
                dep::Fmt (m)
                dep::Fmt (t)
                dep::Fmt (v)
                dep::fmt (t)
                dep::fmt::Display::fmt (a)
            "#]],
        );
    }

    #[test]
    fn search_casing() {
        let ra_fixture = r#"
            //- /main.rs crate:main deps:dep
            //- /dep.rs crate:dep

            pub struct fmt;
            pub struct FMT;
        "#;

        check_search(
            ra_fixture,
            "main",
            Query::new("FMT".to_string()),
            expect![[r#"
                dep::FMT (t)
                dep::FMT (v)
                dep::fmt (t)
                dep::fmt (v)
            "#]],
        );

        check_search(
            ra_fixture,
            "main",
            Query::new("FMT".to_string()).case_sensitive(),
            expect![[r#"
                dep::FMT (t)
                dep::FMT (v)
            "#]],
        );
    }

    #[test]
    fn search_limit() {
        check_search(
            r#"
        //- /main.rs crate:main deps:dep
        //- /dep.rs crate:dep
        pub mod fmt {
            pub trait Display {
                fn fmt();
            }
        }
        #[macro_export]
        macro_rules! Fmt {
            () => {};
        }
        pub struct Fmt;

        pub fn format() {}
        pub fn no() {}
    "#,
            "main",
            Query::new("".to_string()).fuzzy().limit(1),
            expect![[r#"
                dep::fmt::Display (t)
            "#]],
        );
    }
}
