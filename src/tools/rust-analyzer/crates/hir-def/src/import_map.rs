//! A map of all publicly exported items in a crate.

use std::{fmt, hash::BuildHasherDefault};

use base_db::CrateId;
use fst::{self, Streamer};
use hir_expand::name::Name;
use indexmap::{map::Entry, IndexMap};
use itertools::Itertools;
use rustc_hash::{FxHashSet, FxHasher};
use triomphe::Arc;

use crate::{
    db::DefDatabase, item_scope::ItemInNs, nameres::DefMap, visibility::Visibility, AssocItemId,
    ModuleDefId, ModuleId, TraitId,
};

type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

/// Item import details stored in the `ImportMap`.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ImportInfo {
    /// A path that can be used to import the item, relative to the crate's root.
    pub path: ImportPath,
    /// The module containing this item.
    pub container: ModuleId,
    /// Whether the import is a trait associated item or not.
    pub is_trait_assoc_item: bool,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ImportPath {
    pub segments: Vec<Name>,
}

impl ImportPath {
    pub fn display<'a>(&'a self, db: &'a dyn DefDatabase) -> impl fmt::Display + 'a {
        struct Display<'a> {
            db: &'a dyn DefDatabase,
            path: &'a ImportPath,
        }
        impl fmt::Display for Display<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(
                    &self.path.segments.iter().map(|it| it.display(self.db.upcast())).format("::"),
                    f,
                )
            }
        }
        Display { db, path: self }
    }

    fn len(&self) -> usize {
        self.segments.len()
    }
}

/// A map from publicly exported items to the path needed to import/name them from a downstream
/// crate.
///
/// Reexports of items are taken into account, ie. if something is exported under multiple
/// names, the one with the shortest import path will be used.
///
/// Note that all paths are relative to the containing crate's root, so the crate name still needs
/// to be prepended to the `ModPath` before the path is valid.
#[derive(Default)]
pub struct ImportMap {
    map: FxIndexMap<ItemInNs, ImportInfo>,

    /// List of keys stored in `map`, sorted lexicographically by their `ModPath`. Indexed by the
    /// values returned by running `fst`.
    ///
    /// Since a path can refer to multiple items due to namespacing, we store all items with the
    /// same path right after each other. This allows us to find all items after the FST gives us
    /// the index of the first one.
    importables: Vec<ItemInNs>,
    fst: fst::Map<Vec<u8>>,
}

impl ImportMap {
    pub fn import_map_query(db: &dyn DefDatabase, krate: CrateId) -> Arc<Self> {
        let _p = profile::span("import_map_query");

        let mut import_map = collect_import_map(db, krate);

        let mut importables = import_map
            .map
            .iter()
            .map(|(item, info)| (item, fst_path(db, &info.path)))
            .collect::<Vec<_>>();
        importables.sort_by(|(_, fst_path), (_, fst_path2)| fst_path.cmp(fst_path2));

        // Build the FST, taking care not to insert duplicate values.

        let mut builder = fst::MapBuilder::memory();
        let mut last_batch_start = 0;

        for idx in 0..importables.len() {
            let key = &importables[last_batch_start].1;
            if let Some((_, fst_path)) = importables.get(idx + 1) {
                if key == fst_path {
                    continue;
                }
            }

            let _ = builder.insert(key, last_batch_start as u64);

            last_batch_start = idx + 1;
        }

        import_map.fst = builder.into_map();
        import_map.importables = importables.iter().map(|&(&item, _)| item).collect();

        Arc::new(import_map)
    }

    /// Returns the `ModPath` needed to import/mention `item`, relative to this crate's root.
    pub fn path_of(&self, item: ItemInNs) -> Option<&ImportPath> {
        self.import_info_for(item).map(|it| &it.path)
    }

    pub fn import_info_for(&self, item: ItemInNs) -> Option<&ImportInfo> {
        self.map.get(&item)
    }

    #[cfg(test)]
    fn fmt_for_test(&self, db: &dyn DefDatabase) -> String {
        let mut importable_paths: Vec<_> = self
            .map
            .iter()
            .map(|(item, info)| {
                let ns = match item {
                    ItemInNs::Types(_) => "t",
                    ItemInNs::Values(_) => "v",
                    ItemInNs::Macros(_) => "m",
                };
                format!("- {} ({ns})", info.path.display(db))
            })
            .collect();

        importable_paths.sort();
        importable_paths.join("\n")
    }

    fn collect_trait_assoc_items(
        &mut self,
        db: &dyn DefDatabase,
        tr: TraitId,
        is_type_in_ns: bool,
        original_import_info: &ImportInfo,
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

            let mut assoc_item_info = original_import_info.clone();
            assoc_item_info.path.segments.push(assoc_item_name.to_owned());
            assoc_item_info.is_trait_assoc_item = true;
            self.map.insert(assoc_item, assoc_item_info);
        }
    }
}

fn collect_import_map(db: &dyn DefDatabase, krate: CrateId) -> ImportMap {
    let _p = profile::span("collect_import_map");

    let def_map = db.crate_def_map(krate);
    let mut import_map = ImportMap::default();

    // We look only into modules that are public(ly reexported), starting with the crate root.
    let empty = ImportPath { segments: vec![] };
    let root = def_map.module_id(DefMap::ROOT);
    let mut worklist = vec![(root, empty)];
    while let Some((module, mod_path)) = worklist.pop() {
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
            let mk_path = || {
                let mut path = mod_path.clone();
                path.segments.push(name.clone());
                path
            };

            for item in per_ns.iter_items() {
                let path = mk_path();
                let path_len = path.len();
                let import_info =
                    ImportInfo { path, container: module, is_trait_assoc_item: false };

                if let Some(ModuleDefId::TraitId(tr)) = item.as_module_def_id() {
                    import_map.collect_trait_assoc_items(
                        db,
                        tr,
                        matches!(item, ItemInNs::Types(_)),
                        &import_info,
                    );
                }

                match import_map.map.entry(item) {
                    Entry::Vacant(entry) => {
                        entry.insert(import_info);
                    }
                    Entry::Occupied(mut entry) => {
                        // If the new path is shorter, prefer that one.
                        if path_len < entry.get().path.len() {
                            *entry.get_mut() = import_info;
                        } else {
                            continue;
                        }
                    }
                }

                // If we've just added a path to a module, descend into it. We might traverse
                // modules multiple times, but only if the new path to it is shorter than the
                // first (else we `continue` above).
                if let Some(ModuleDefId::ModuleId(mod_id)) = item.as_module_def_id() {
                    worklist.push((mod_id, mk_path()));
                }
            }
        }
    }

    import_map
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
        let mut importable_paths: Vec<_> = self
            .map
            .iter()
            .map(|(item, _)| match item {
                ItemInNs::Types(it) => format!("- {it:?} (t)",),
                ItemInNs::Values(it) => format!("- {it:?} (v)",),
                ItemInNs::Macros(it) => format!("- {it:?} (m)",),
            })
            .collect();

        importable_paths.sort();
        f.write_str(&importable_paths.join("\n"))
    }
}

fn fst_path(db: &dyn DefDatabase, path: &ImportPath) -> String {
    let _p = profile::span("fst_path");
    let mut s = path.display(db).to_string();
    s.make_ascii_lowercase();
    s
}

#[derive(Debug, Eq, PartialEq, Hash)]
pub enum ImportKind {
    Module,
    Function,
    Adt,
    EnumVariant,
    Const,
    Static,
    Trait,
    TraitAlias,
    TypeAlias,
    BuiltinType,
    AssociatedItem,
    Macro,
}

/// A way to match import map contents against the search query.
#[derive(Debug)]
pub enum SearchMode {
    /// Import map entry should strictly match the query string.
    Equals,
    /// Import map entry should contain the query string.
    Contains,
    /// Import map entry should contain all letters from the query string,
    /// in the same order, but not necessary adjacent.
    Fuzzy,
}

#[derive(Debug)]
pub struct Query {
    query: String,
    lowercased: String,
    name_only: bool,
    assoc_items_only: bool,
    search_mode: SearchMode,
    case_sensitive: bool,
    limit: usize,
    exclude_import_kinds: FxHashSet<ImportKind>,
}

impl Query {
    pub fn new(query: String) -> Self {
        let lowercased = query.to_lowercase();
        Self {
            query,
            lowercased,
            name_only: false,
            assoc_items_only: false,
            search_mode: SearchMode::Contains,
            case_sensitive: false,
            limit: usize::max_value(),
            exclude_import_kinds: FxHashSet::default(),
        }
    }

    /// Matches entries' names only, ignoring the rest of
    /// the qualifier.
    /// Example: for `std::marker::PhantomData`, the name is `PhantomData`.
    pub fn name_only(self) -> Self {
        Self { name_only: true, ..self }
    }

    /// Matches only the entries that are associated items, ignoring the rest.
    pub fn assoc_items_only(self) -> Self {
        Self { assoc_items_only: true, ..self }
    }

    /// Specifies the way to search for the entries using the query.
    pub fn search_mode(self, search_mode: SearchMode) -> Self {
        Self { search_mode, ..self }
    }

    /// Limits the returned number of items to `limit`.
    pub fn limit(self, limit: usize) -> Self {
        Self { limit, ..self }
    }

    /// Respect casing of the query string when matching.
    pub fn case_sensitive(self) -> Self {
        Self { case_sensitive: true, ..self }
    }

    /// Do not include imports of the specified kind in the search results.
    pub fn exclude_import_kind(mut self, import_kind: ImportKind) -> Self {
        self.exclude_import_kinds.insert(import_kind);
        self
    }

    fn import_matches(
        &self,
        db: &dyn DefDatabase,
        import: &ImportInfo,
        enforce_lowercase: bool,
    ) -> bool {
        let _p = profile::span("import_map::Query::import_matches");
        if import.is_trait_assoc_item {
            if self.exclude_import_kinds.contains(&ImportKind::AssociatedItem) {
                return false;
            }
        } else if self.assoc_items_only {
            return false;
        }

        let mut input = if import.is_trait_assoc_item || self.name_only {
            import.path.segments.last().unwrap().display(db.upcast()).to_string()
        } else {
            import.path.display(db).to_string()
        };
        if enforce_lowercase || !self.case_sensitive {
            input.make_ascii_lowercase();
        }

        let query_string =
            if !enforce_lowercase && self.case_sensitive { &self.query } else { &self.lowercased };

        match self.search_mode {
            SearchMode::Equals => &input == query_string,
            SearchMode::Contains => input.contains(query_string),
            SearchMode::Fuzzy => {
                let mut unchecked_query_chars = query_string.chars();
                let mut mismatching_query_char = unchecked_query_chars.next();

                for input_char in input.chars() {
                    match mismatching_query_char {
                        None => return true,
                        Some(matching_query_char) if matching_query_char == input_char => {
                            mismatching_query_char = unchecked_query_chars.next();
                        }
                        _ => (),
                    }
                }
                mismatching_query_char.is_none()
            }
        }
    }
}

/// Searches dependencies of `krate` for an importable path matching `query`.
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

    let mut all_indexed_values = FxHashSet::default();
    while let Some((_, indexed_values)) = stream.next() {
        all_indexed_values.extend(indexed_values.iter().copied());
    }

    let mut res = FxHashSet::default();
    for indexed_value in all_indexed_values {
        let import_map = &import_maps[indexed_value.index];
        let importables = &import_map.importables[indexed_value.value as usize..];

        let common_importable_data = &import_map.map[&importables[0]];
        if !query.import_matches(db, common_importable_data, true) {
            continue;
        }

        // Path shared by the importable items in this group.
        let common_importables_path_fst = fst_path(db, &common_importable_data.path);
        // Add the items from this `ModPath` group. Those are all subsequent items in
        // `importables` whose paths match `path`.
        let iter = importables
            .iter()
            .copied()
            .take_while(|item| {
                common_importables_path_fst == fst_path(db, &import_map.map[item].path)
            })
            .filter(|&item| match item_import_kind(item) {
                Some(import_kind) => !query.exclude_import_kinds.contains(&import_kind),
                None => true,
            })
            .filter(|item| {
                !query.case_sensitive // we've already checked the common importables path case-insensitively
                        || query.import_matches(db, &import_map.map[item], false)
            });
        res.extend(iter);

        if res.len() >= query.limit {
            return res;
        }
    }

    res
}

fn item_import_kind(item: ItemInNs) -> Option<ImportKind> {
    Some(match item.as_module_def_id()? {
        ModuleDefId::ModuleId(_) => ImportKind::Module,
        ModuleDefId::FunctionId(_) => ImportKind::Function,
        ModuleDefId::AdtId(_) => ImportKind::Adt,
        ModuleDefId::EnumVariantId(_) => ImportKind::EnumVariant,
        ModuleDefId::ConstId(_) => ImportKind::Const,
        ModuleDefId::StaticId(_) => ImportKind::Static,
        ModuleDefId::TraitId(_) => ImportKind::Trait,
        ModuleDefId::TraitAliasId(_) => ImportKind::TraitAlias,
        ModuleDefId::TypeAliasId(_) => ImportKind::TypeAlias,
        ModuleDefId::BuiltinType(_) => ImportKind::BuiltinType,
        ModuleDefId::MacroId(_) => ImportKind::Macro,
    })
}

#[cfg(test)]
mod tests {
    use base_db::{fixture::WithFixture, SourceDatabase, Upcast};
    use expect_test::{expect, Expect};

    use crate::{db::DefDatabase, test_db::TestDB, ItemContainerId, Lookup};

    use super::*;

    fn check_search(ra_fixture: &str, crate_name: &str, query: Query, expect: Expect) {
        let db = TestDB::with_files(ra_fixture);
        let crate_graph = db.crate_graph();
        let krate = crate_graph
            .iter()
            .find(|krate| {
                crate_graph[*krate].display_name.as_ref().map(|n| n.to_string())
                    == Some(crate_name.to_string())
            })
            .unwrap();

        let actual = search_dependencies(db.upcast(), krate, query)
            .into_iter()
            .filter_map(|dependency| {
                let dependency_krate = dependency.krate(db.upcast())?;
                let dependency_imports = db.import_map(dependency_krate);

                let (path, mark) = match assoc_item_path(&db, &dependency_imports, dependency) {
                    Some(assoc_item_path) => (assoc_item_path, "a"),
                    None => (
                        dependency_imports.path_of(dependency)?.display(&db).to_string(),
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
        let dependency_assoc_item_id = match dependency {
            ItemInNs::Types(ModuleDefId::FunctionId(id))
            | ItemInNs::Values(ModuleDefId::FunctionId(id)) => AssocItemId::from(id),
            ItemInNs::Types(ModuleDefId::ConstId(id))
            | ItemInNs::Values(ModuleDefId::ConstId(id)) => AssocItemId::from(id),
            ItemInNs::Types(ModuleDefId::TypeAliasId(id))
            | ItemInNs::Values(ModuleDefId::TypeAliasId(id)) => AssocItemId::from(id),
            _ => return None,
        };

        let trait_ = assoc_to_trait(db, dependency)?;
        if let ModuleDefId::TraitId(tr) = trait_.as_module_def_id()? {
            let trait_data = db.trait_data(tr);
            let assoc_item_name =
                trait_data.items.iter().find_map(|(assoc_item_name, assoc_item_id)| {
                    if &dependency_assoc_item_id == assoc_item_id {
                        Some(assoc_item_name)
                    } else {
                        None
                    }
                })?;
            return Some(format!(
                "{}::{}",
                dependency_imports.path_of(trait_)?.display(db),
                assoc_item_name.display(db.upcast())
            ));
        }
        None
    }

    fn assoc_to_trait(db: &dyn DefDatabase, item: ItemInNs) -> Option<ItemInNs> {
        let assoc: AssocItemId = match item {
            ItemInNs::Types(it) | ItemInNs::Values(it) => match it {
                ModuleDefId::TypeAliasId(it) => it.into(),
                ModuleDefId::FunctionId(it) => it.into(),
                ModuleDefId::ConstId(it) => it.into(),
                _ => return None,
            },
            _ => return None,
        };

        let container = match assoc {
            AssocItemId::FunctionId(it) => it.lookup(db).container,
            AssocItemId::ConstId(it) => it.lookup(db).container,
            AssocItemId::TypeAliasId(it) => it.lookup(db).container,
        };

        match container {
            ItemContainerId::TraitId(it) => Some(ItemInNs::Types(it.into())),
            _ => None,
        }
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
                - reexported_module (t)
                - reexported_module::S (t)
                - reexported_module::S (v)
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
            Query::new("fmt".to_string()).search_mode(SearchMode::Fuzzy),
            expect![[r#"
                dep::fmt (t)
                dep::fmt::Display (t)
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
            Query::new("fmt".to_string()).search_mode(SearchMode::Fuzzy).assoc_items_only(),
            expect![[r#"
                dep::fmt::Display::FMT_CONST (a)
                dep::fmt::Display::format_function (a)
                dep::fmt::Display::format_method (a)
            "#]],
        );

        check_search(
            ra_fixture,
            "main",
            Query::new("fmt".to_string())
                .search_mode(SearchMode::Fuzzy)
                .exclude_import_kind(ImportKind::AssociatedItem),
            expect![[r#"
            dep::fmt (t)
            dep::fmt::Display (t)
        "#]],
        );

        check_search(
            ra_fixture,
            "main",
            Query::new("fmt".to_string())
                .search_mode(SearchMode::Fuzzy)
                .assoc_items_only()
                .exclude_import_kind(ImportKind::AssociatedItem),
            expect![[r#""#]],
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
            Query::new("fmt".to_string()).search_mode(SearchMode::Fuzzy),
            expect![[r#"
                dep::Fmt (m)
                dep::Fmt (t)
                dep::Fmt (v)
                dep::fmt (t)
                dep::fmt::Display (t)
                dep::fmt::Display::fmt (a)
                dep::format (f)
            "#]],
        );

        check_search(
            ra_fixture,
            "main",
            Query::new("fmt".to_string()).search_mode(SearchMode::Equals),
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
            Query::new("fmt".to_string()).search_mode(SearchMode::Contains),
            expect![[r#"
                dep::Fmt (m)
                dep::Fmt (t)
                dep::Fmt (v)
                dep::fmt (t)
                dep::fmt::Display (t)
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
                dep::fmt::Display (t)
                dep::fmt::Display::fmt (a)
            "#]],
        );

        check_search(
            ra_fixture,
            "main",
            Query::new("fmt".to_string()).name_only(),
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
            Query::new("".to_string()).limit(2),
            expect![[r#"
                dep::Fmt (m)
                dep::Fmt (t)
                dep::Fmt (v)
                dep::fmt (t)
            "#]],
        );
    }

    #[test]
    fn search_exclusions() {
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
            Query::new("FMT".to_string()).exclude_import_kind(ImportKind::Adt),
            expect![[r#""#]],
        );
    }
}
