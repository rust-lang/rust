//! A map of all publicly exported items in a crate.

use std::collections::hash_map::Entry;
use std::{fmt, hash::BuildHasherDefault};

use base_db::CrateId;
use fst::{self, Streamer};
use hir_expand::name::Name;
use indexmap::IndexMap;
use itertools::Itertools;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use triomphe::Arc;

use crate::{
    db::DefDatabase, item_scope::ItemInNs, nameres::DefMap, visibility::Visibility, AssocItemId,
    ModuleDefId, ModuleId, TraitId,
};

type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

// FIXME: Support aliases: an item may be exported under multiple names, so `ImportInfo` should
// have `Vec<(Name, ModuleId)>` instead of `(Name, ModuleId)`.
/// Item import details stored in the `ImportMap`.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ImportInfo {
    /// A name that can be used to import the item, relative to the crate's root.
    pub name: Name,
    /// The module containing this item.
    pub container: ModuleId,
    /// Whether the import is a trait associated item or not.
    pub is_trait_assoc_item: bool,
}

/// A map from publicly exported items to its name.
///
/// Reexports of items are taken into account, ie. if something is exported under multiple
/// names, the one with the shortest import path will be used.
#[derive(Default)]
pub struct ImportMap {
    map: FxIndexMap<ItemInNs, ImportInfo>,

    /// List of keys stored in `map`, sorted lexicographically by their `ModPath`. Indexed by the
    /// values returned by running `fst`.
    ///
    /// Since a name can refer to multiple items due to namespacing, we store all items with the
    /// same name right after each other. This allows us to find all items after the FST gives us
    /// the index of the first one.
    importables: Vec<ItemInNs>,
    fst: fst::Map<Vec<u8>>,
}

impl ImportMap {
    pub(crate) fn import_map_query(db: &dyn DefDatabase, krate: CrateId) -> Arc<Self> {
        let _p = profile::span("import_map_query");

        let map = collect_import_map(db, krate);

        let mut importables: Vec<_> = map
            .iter()
            // We've only collected items, whose name cannot be tuple field.
            .map(|(&item, info)| (item, info.name.as_str().unwrap().to_ascii_lowercase()))
            .collect();
        importables.sort_by(|(_, lhs_name), (_, rhs_name)| lhs_name.cmp(rhs_name));

        // Build the FST, taking care not to insert duplicate values.
        let mut builder = fst::MapBuilder::memory();
        let iter = importables.iter().enumerate().dedup_by(|lhs, rhs| lhs.1 .1 == rhs.1 .1);
        for (start_idx, (_, name)) in iter {
            let _ = builder.insert(name, start_idx as u64);
        }

        Arc::new(ImportMap {
            map,
            fst: builder.into_map(),
            importables: importables.into_iter().map(|(item, _)| item).collect(),
        })
    }

    pub fn import_info_for(&self, item: ItemInNs) -> Option<&ImportInfo> {
        self.map.get(&item)
    }
}

fn collect_import_map(db: &dyn DefDatabase, krate: CrateId) -> FxIndexMap<ItemInNs, ImportInfo> {
    let _p = profile::span("collect_import_map");

    let def_map = db.crate_def_map(krate);
    let mut map = FxIndexMap::default();

    // We look only into modules that are public(ly reexported), starting with the crate root.
    let root = def_map.module_id(DefMap::ROOT);
    let mut worklist = vec![(root, 0)];
    // Records items' minimum module depth.
    let mut depth_map = FxHashMap::default();

    while let Some((module, depth)) = worklist.pop() {
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
            for item in per_ns.iter_items() {
                let import_info = ImportInfo {
                    name: name.clone(),
                    container: module,
                    is_trait_assoc_item: false,
                };

                match depth_map.entry(item) {
                    Entry::Vacant(entry) => {
                        entry.insert(depth);
                    }
                    Entry::Occupied(mut entry) => {
                        if depth < *entry.get() {
                            entry.insert(depth);
                        } else {
                            continue;
                        }
                    }
                }

                if let Some(ModuleDefId::TraitId(tr)) = item.as_module_def_id() {
                    collect_trait_assoc_items(
                        db,
                        &mut map,
                        tr,
                        matches!(item, ItemInNs::Types(_)),
                        &import_info,
                    );
                }

                map.insert(item, import_info);

                // If we've just added a module, descend into it. We might traverse modules
                // multiple times, but only if the module depth is smaller (else we `continue`
                // above).
                if let Some(ModuleDefId::ModuleId(mod_id)) = item.as_module_def_id() {
                    worklist.push((mod_id, depth + 1));
                }
            }
        }
    }

    map
}

fn collect_trait_assoc_items(
    db: &dyn DefDatabase,
    map: &mut FxIndexMap<ItemInNs, ImportInfo>,
    tr: TraitId,
    is_type_in_ns: bool,
    trait_import_info: &ImportInfo,
) {
    let _p = profile::span("collect_trait_assoc_items");
    for (assoc_item_name, item) in &db.trait_data(tr).items {
        let module_def_id = match item {
            AssocItemId::FunctionId(f) => ModuleDefId::from(*f),
            AssocItemId::ConstId(c) => ModuleDefId::from(*c),
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

        let assoc_item_info = ImportInfo {
            container: trait_import_info.container,
            name: assoc_item_name.clone(),
            is_trait_assoc_item: true,
        };
        map.insert(assoc_item, assoc_item_info);
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
            .map(|(item, _)| match item {
                ItemInNs::Types(it) => format!("- {it:?} (t)",),
                ItemInNs::Values(it) => format!("- {it:?} (v)",),
                ItemInNs::Macros(it) => format!("- {it:?} (m)",),
            })
            .collect();

        importable_names.sort();
        f.write_str(&importable_names.join("\n"))
    }
}

/// A way to match import map contents against the search query.
#[derive(Debug)]
enum SearchMode {
    /// Import map entry should strictly match the query string.
    Exact,
    /// Import map entry should contain all letters from the query string,
    /// in the same order, but not necessary adjacent.
    Fuzzy,
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

    fn import_matches(
        &self,
        db: &dyn DefDatabase,
        import: &ImportInfo,
        enforce_lowercase: bool,
    ) -> bool {
        let _p = profile::span("import_map::Query::import_matches");
        match (import.is_trait_assoc_item, self.assoc_mode) {
            (true, AssocSearchMode::Exclude) => return false,
            (false, AssocSearchMode::AssocItemsOnly) => return false,
            _ => {}
        }

        let mut input = import.name.display(db.upcast()).to_string();
        let case_insensitive = enforce_lowercase || !self.case_sensitive;
        if case_insensitive {
            input.make_ascii_lowercase();
        }

        let query_string = if case_insensitive { &self.lowercased } else { &self.query };

        match self.search_mode {
            SearchMode::Exact => &input == query_string,
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
    query: Query,
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
    while let Some((_, indexed_values)) = stream.next() {
        for indexed_value in indexed_values {
            let import_map = &import_maps[indexed_value.index];
            let importables = &import_map.importables[indexed_value.value as usize..];

            let common_importable_data = &import_map.map[&importables[0]];
            if !query.import_matches(db, common_importable_data, true) {
                continue;
            }

            // Name shared by the importable items in this group.
            let common_importable_name =
                common_importable_data.name.to_smol_str().to_ascii_lowercase();
            // Add the items from this name group. Those are all subsequent items in
            // `importables` whose name match `common_importable_name`.
            let iter = importables
                .iter()
                .copied()
                .take_while(|item| {
                    common_importable_name
                        == import_map.map[item].name.to_smol_str().to_ascii_lowercase()
                })
                .filter(|item| {
                    !query.case_sensitive // we've already checked the common importables name case-insensitively
                        || query.import_matches(db, &import_map.map[item], false)
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
                        render_path(&db, dependency_imports.import_info_for(dependency)?),
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
        Some(format!("{}::{}", render_path(db, trait_info), assoc_item_name.display(db.upcast())))
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
                - sub (t)
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
