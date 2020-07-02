//! A map of all publicly exported items in a crate.

use std::{cmp::Ordering, fmt, hash::BuildHasherDefault, sync::Arc};

use fst::{self, Streamer};
use indexmap::{map::Entry, IndexMap};
use ra_db::CrateId;
use ra_syntax::SmolStr;
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;

use crate::{
    db::DefDatabase,
    item_scope::ItemInNs,
    path::{ModPath, PathKind},
    visibility::Visibility,
    AssocItemId, ModuleDefId, ModuleId, TraitId,
};

type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

/// Item import details stored in the `ImportMap`.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ImportInfo {
    /// A path that can be used to import the item, relative to the crate's root.
    pub path: ModPath,
    /// The module containing this item.
    pub container: ModuleId,
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

    /// Maps names of associated items to the item's ID. Only includes items whose defining trait is
    /// exported.
    assoc_map: FxHashMap<SmolStr, SmallVec<[AssocItemId; 1]>>,
}

impl ImportMap {
    pub fn import_map_query(db: &dyn DefDatabase, krate: CrateId) -> Arc<Self> {
        let _p = ra_prof::profile("import_map_query");
        let def_map = db.crate_def_map(krate);
        let mut import_map = Self::default();

        // We look only into modules that are public(ly reexported), starting with the crate root.
        let empty = ModPath { kind: PathKind::Plain, segments: vec![] };
        let root = ModuleId { krate, local_id: def_map.root };
        let mut worklist = vec![(root, empty)];
        while let Some((module, mod_path)) = worklist.pop() {
            let ext_def_map;
            let mod_data = if module.krate == krate {
                &def_map[module.local_id]
            } else {
                // The crate might reexport a module defined in another crate.
                ext_def_map = db.crate_def_map(module.krate);
                &ext_def_map[module.local_id]
            };

            let visible_items = mod_data.scope.entries().filter_map(|(name, per_ns)| {
                let per_ns = per_ns.filter_visibility(|vis| vis == Visibility::Public);
                if per_ns.is_none() {
                    None
                } else {
                    Some((name, per_ns))
                }
            });

            for (name, per_ns) in visible_items {
                let mk_path = || {
                    let mut path = mod_path.clone();
                    path.segments.push(name.clone());
                    path
                };

                for item in per_ns.iter_items() {
                    let path = mk_path();
                    match import_map.map.entry(item) {
                        Entry::Vacant(entry) => {
                            entry.insert(ImportInfo { path, container: module });
                        }
                        Entry::Occupied(mut entry) => {
                            // If the new path is shorter, prefer that one.
                            if path.len() < entry.get().path.len() {
                                *entry.get_mut() = ImportInfo { path, container: module };
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

                    // If we've added a path to a trait, add the trait's methods to the method map.
                    if let Some(ModuleDefId::TraitId(tr)) = item.as_module_def_id() {
                        import_map.collect_trait_methods(db, tr);
                    }
                }
            }
        }

        let mut importables = import_map.map.iter().collect::<Vec<_>>();

        importables.sort_by(cmp);

        // Build the FST, taking care not to insert duplicate values.

        let mut builder = fst::MapBuilder::memory();
        let mut last_batch_start = 0;

        for idx in 0..importables.len() {
            if let Some(next_item) = importables.get(idx + 1) {
                if cmp(&importables[last_batch_start], next_item) == Ordering::Equal {
                    continue;
                }
            }

            let start = last_batch_start;
            last_batch_start = idx + 1;

            let key = fst_path(&importables[start].1.path);

            builder.insert(key, start as u64).unwrap();
        }

        import_map.fst = fst::Map::new(builder.into_inner().unwrap()).unwrap();
        import_map.importables = importables.iter().map(|(item, _)| **item).collect();

        Arc::new(import_map)
    }

    /// Returns the `ModPath` needed to import/mention `item`, relative to this crate's root.
    pub fn path_of(&self, item: ItemInNs) -> Option<&ModPath> {
        Some(&self.map.get(&item)?.path)
    }

    pub fn import_info_for(&self, item: ItemInNs) -> Option<&ImportInfo> {
        self.map.get(&item)
    }

    fn collect_trait_methods(&mut self, db: &dyn DefDatabase, tr: TraitId) {
        let data = db.trait_data(tr);
        for (name, item) in data.items.iter() {
            self.assoc_map.entry(name.to_string().into()).or_default().push(*item);
        }
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
        let mut importable_paths: Vec<_> = self
            .map
            .iter()
            .map(|(item, info)| {
                let ns = match item {
                    ItemInNs::Types(_) => "t",
                    ItemInNs::Values(_) => "v",
                    ItemInNs::Macros(_) => "m",
                };
                format!("- {} ({})", info.path, ns)
            })
            .collect();

        importable_paths.sort();
        f.write_str(&importable_paths.join("\n"))
    }
}

fn fst_path(path: &ModPath) -> String {
    let mut s = path.to_string();
    s.make_ascii_lowercase();
    s
}

fn cmp((_, lhs): &(&ItemInNs, &ImportInfo), (_, rhs): &(&ItemInNs, &ImportInfo)) -> Ordering {
    let lhs_str = fst_path(&lhs.path);
    let rhs_str = fst_path(&rhs.path);
    lhs_str.cmp(&rhs_str)
}

#[derive(Debug)]
pub struct Query {
    query: String,
    lowercased: String,
    anchor_end: bool,
    case_sensitive: bool,
    limit: usize,
}

impl Query {
    pub fn new(query: &str) -> Self {
        Self {
            lowercased: query.to_lowercase(),
            query: query.to_string(),
            anchor_end: false,
            case_sensitive: false,
            limit: usize::max_value(),
        }
    }

    /// Only returns items whose paths end with the (case-insensitive) query string as their last
    /// segment.
    pub fn anchor_end(self) -> Self {
        Self { anchor_end: true, ..self }
    }

    /// Limits the returned number of items to `limit`.
    pub fn limit(self, limit: usize) -> Self {
        Self { limit, ..self }
    }

    /// Respect casing of the query string when matching.
    pub fn case_sensitive(self) -> Self {
        Self { case_sensitive: true, ..self }
    }
}

/// Searches dependencies of `krate` for an importable path matching `query`.
///
/// This returns a list of items that could be imported from dependencies of `krate`.
pub fn search_dependencies<'a>(
    db: &'a dyn DefDatabase,
    krate: CrateId,
    query: Query,
) -> Vec<ItemInNs> {
    let _p = ra_prof::profile("search_dependencies").detail(|| format!("{:?}", query));

    let graph = db.crate_graph();
    let import_maps: Vec<_> =
        graph[krate].dependencies.iter().map(|dep| db.import_map(dep.crate_id)).collect();

    let automaton = fst::automaton::Subsequence::new(&query.lowercased);

    let mut op = fst::map::OpBuilder::new();
    for map in &import_maps {
        op = op.add(map.fst.search(&automaton));
    }

    let mut stream = op.union();
    let mut res = Vec::new();
    while let Some((_, indexed_values)) = stream.next() {
        for indexed_value in indexed_values {
            let import_map = &import_maps[indexed_value.index];
            let importables = &import_map.importables[indexed_value.value as usize..];

            // Path shared by the importable items in this group.
            let path = &import_map.map[&importables[0]].path;

            if query.anchor_end {
                // Last segment must match query.
                let last = path.segments.last().unwrap().to_string();
                if last.to_lowercase() != query.lowercased {
                    continue;
                }
            }

            // Add the items from this `ModPath` group. Those are all subsequent items in
            // `importables` whose paths match `path`.
            let iter = importables.iter().copied().take_while(|item| {
                let item_path = &import_map.map[item].path;
                fst_path(item_path) == fst_path(path)
            });

            if query.case_sensitive {
                // FIXME: This does not do a subsequence match.
                res.extend(iter.filter(|item| {
                    let item_path = &import_map.map[item].path;
                    item_path.to_string().contains(&query.query)
                }));
            } else {
                res.extend(iter);
            }

            if res.len() >= query.limit {
                res.truncate(query.limit);
                return res;
            }
        }
    }

    // Add all exported associated items whose names match the query (exactly).
    for map in &import_maps {
        if let Some(v) = map.assoc_map.get(&*query.query) {
            res.extend(v.iter().map(|&assoc| {
                ItemInNs::Types(match assoc {
                    AssocItemId::FunctionId(it) => it.into(),
                    AssocItemId::ConstId(it) => it.into(),
                    AssocItemId::TypeAliasId(it) => it.into(),
                })
            }));
        }
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_db::TestDB, AssocContainerId, Lookup};
    use insta::assert_snapshot;
    use itertools::Itertools;
    use ra_db::fixture::WithFixture;
    use ra_db::{SourceDatabase, Upcast};

    fn import_map(ra_fixture: &str) -> String {
        let db = TestDB::with_files(ra_fixture);
        let crate_graph = db.crate_graph();

        let s = crate_graph
            .iter()
            .filter_map(|krate| {
                let cdata = &crate_graph[krate];
                let name = cdata.display_name.as_ref()?;

                let map = db.import_map(krate);

                Some(format!("{}:\n{:?}", name, map))
            })
            .join("\n");
        s
    }

    fn search_dependencies_of(ra_fixture: &str, krate_name: &str, query: Query) -> String {
        let db = TestDB::with_files(ra_fixture);
        let crate_graph = db.crate_graph();
        let krate = crate_graph
            .iter()
            .find(|krate| {
                crate_graph[*krate].display_name.as_ref().map(|n| n.to_string())
                    == Some(krate_name.to_string())
            })
            .unwrap();

        search_dependencies(db.upcast(), krate, query)
            .into_iter()
            .filter_map(|item| {
                let mark = match item {
                    ItemInNs::Types(_) => "t",
                    ItemInNs::Values(_) => "v",
                    ItemInNs::Macros(_) => "m",
                };
                let item = assoc_to_trait(&db, item);
                item.krate(db.upcast()).map(|krate| {
                    let map = db.import_map(krate);
                    let path = map.path_of(item).unwrap();
                    format!(
                        "{}::{} ({})",
                        crate_graph[krate].display_name.as_ref().unwrap(),
                        path,
                        mark
                    )
                })
            })
            .join("\n")
    }

    fn assoc_to_trait(db: &dyn DefDatabase, item: ItemInNs) -> ItemInNs {
        let assoc: AssocItemId = match item {
            ItemInNs::Types(it) | ItemInNs::Values(it) => match it {
                ModuleDefId::TypeAliasId(it) => it.into(),
                ModuleDefId::FunctionId(it) => it.into(),
                ModuleDefId::ConstId(it) => it.into(),
                _ => return item,
            },
            _ => return item,
        };

        let container = match assoc {
            AssocItemId::FunctionId(it) => it.lookup(db).container,
            AssocItemId::ConstId(it) => it.lookup(db).container,
            AssocItemId::TypeAliasId(it) => it.lookup(db).container,
        };

        match container {
            AssocContainerId::TraitId(it) => ItemInNs::Types(it.into()),
            _ => item,
        }
    }

    #[test]
    fn smoke() {
        let map = import_map(
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
        );

        assert_snapshot!(map, @r###"
        main:
        - publ1 (t)
        - real_pu2 (t)
        - real_pub (t)
        - real_pub::Pub (t)
        lib:
        - Pub (t)
        - Pub2 (t)
        - Pub2 (v)
        "###);
    }

    #[test]
    fn prefers_shortest_path() {
        let map = import_map(
            r"
            //- /main.rs crate:main

            pub mod sub {
                pub mod subsub {
                    pub struct Def {}
                }

                pub use super::sub::subsub::Def;
            }
        ",
        );

        assert_snapshot!(map, @r###"
        main:
        - sub (t)
        - sub::Def (t)
        - sub::subsub (t)
        "###);
    }

    #[test]
    fn type_reexport_cross_crate() {
        // Reexports need to be visible from a crate, even if the original crate exports the item
        // at a shorter path.
        let map = import_map(
            r"
            //- /main.rs crate:main deps:lib
            pub mod m {
                pub use lib::S;
            }
            //- /lib.rs crate:lib
            pub struct S;
        ",
        );

        assert_snapshot!(map, @r###"
        main:
        - m (t)
        - m::S (t)
        - m::S (v)
        lib:
        - S (t)
        - S (v)
        "###);
    }

    #[test]
    fn macro_reexport() {
        let map = import_map(
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
        );

        assert_snapshot!(map, @r###"
        main:
        - m (t)
        - m::pub_macro (m)
        lib:
        - pub_macro (m)
        "###);
    }

    #[test]
    fn module_reexport() {
        // Reexporting modules from a dependency adds all contents to the import map.
        let map = import_map(
            r"
            //- /main.rs crate:main deps:lib
            pub use lib::module as reexported_module;
            //- /lib.rs crate:lib
            pub mod module {
                pub struct S;
            }
        ",
        );

        assert_snapshot!(map, @r###"
        main:
        - reexported_module (t)
        - reexported_module::S (t)
        - reexported_module::S (v)
        lib:
        - module (t)
        - module::S (t)
        - module::S (v)
        "###);
    }

    #[test]
    fn cyclic_module_reexport() {
        // A cyclic reexport does not hang.
        let map = import_map(
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
        );

        assert_snapshot!(map, @r###"
        lib:
        - module (t)
        - module::S (t)
        - module::S (v)
        - sub (t)
        "###);
    }

    #[test]
    fn private_macro() {
        let map = import_map(
            r"
            //- /lib.rs crate:lib
            macro_rules! private_macro {
                () => {};
            }
        ",
        );

        assert_snapshot!(map, @r###"
        lib:
        "###);
    }

    #[test]
    fn namespacing() {
        let map = import_map(
            r"
            //- /lib.rs crate:lib
            pub struct Thing;     // t + v
            #[macro_export]
            macro_rules! Thing {  // m
                () => {};
            }
        ",
        );

        assert_snapshot!(map, @r###"
        lib:
        - Thing (m)
        - Thing (t)
        - Thing (v)
        "###);

        let map = import_map(
            r"
            //- /lib.rs crate:lib
            pub mod Thing {}      // t
            #[macro_export]
            macro_rules! Thing {  // m
                () => {};
            }
        ",
        );

        assert_snapshot!(map, @r###"
        lib:
        - Thing (m)
        - Thing (t)
        "###);
    }

    #[test]
    fn search() {
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

        let res = search_dependencies_of(ra_fixture, "main", Query::new("fmt"));
        assert_snapshot!(res, @r###"
        dep::fmt (t)
        dep::Fmt (t)
        dep::Fmt (v)
        dep::Fmt (m)
        dep::fmt::Display (t)
        dep::format (v)
        dep::fmt::Display (t)
        "###);

        let res = search_dependencies_of(ra_fixture, "main", Query::new("fmt").anchor_end());
        assert_snapshot!(res, @r###"
        dep::fmt (t)
        dep::Fmt (t)
        dep::Fmt (v)
        dep::Fmt (m)
        dep::fmt::Display (t)
        "###);
    }

    #[test]
    fn search_casing() {
        let ra_fixture = r#"
            //- /main.rs crate:main deps:dep
            //- /dep.rs crate:dep

            pub struct fmt;
            pub struct FMT;
        "#;

        let res = search_dependencies_of(ra_fixture, "main", Query::new("FMT"));

        assert_snapshot!(res, @r###"
        dep::fmt (t)
        dep::fmt (v)
        dep::FMT (t)
        dep::FMT (v)
        "###);

        let res = search_dependencies_of(ra_fixture, "main", Query::new("FMT").case_sensitive());

        assert_snapshot!(res, @r###"
        dep::FMT (t)
        dep::FMT (v)
        "###);
    }

    #[test]
    fn search_limit() {
        let res = search_dependencies_of(
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
            Query::new("").limit(2),
        );
        assert_snapshot!(res, @r###"
        dep::fmt (t)
        dep::Fmt (t)
        "###);
    }
}
