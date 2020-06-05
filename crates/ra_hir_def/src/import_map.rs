//! A map of all publicly exported items in a crate.

use std::{collections::hash_map::Entry, fmt, sync::Arc};

use ra_db::CrateId;
use rustc_hash::FxHashMap;

use crate::{
    db::DefDatabase,
    item_scope::ItemInNs,
    path::{ModPath, PathKind},
    visibility::Visibility,
    ModuleDefId, ModuleId,
};

/// A map from publicly exported items to the path needed to import/name them from a downstream
/// crate.
///
/// Reexports of items are taken into account, ie. if something is exported under multiple
/// names, the one with the shortest import path will be used.
///
/// Note that all paths are relative to the containing crate's root, so the crate name still needs
/// to be prepended to the `ModPath` before the path is valid.
#[derive(Eq, PartialEq)]
pub struct ImportMap {
    map: FxHashMap<ItemInNs, ModPath>,
}

impl ImportMap {
    pub fn import_map_query(db: &dyn DefDatabase, krate: CrateId) -> Arc<Self> {
        let _p = ra_prof::profile("import_map_query");
        let def_map = db.crate_def_map(krate);
        let mut import_map = FxHashMap::with_capacity_and_hasher(64, Default::default());

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
                    match import_map.entry(item) {
                        Entry::Vacant(entry) => {
                            entry.insert(path);
                        }
                        Entry::Occupied(mut entry) => {
                            // If the new path is shorter, prefer that one.
                            if path.len() < entry.get().len() {
                                *entry.get_mut() = path;
                            } else {
                                continue;
                            }
                        }
                    }

                    // If we've just added a path to a module, descend into it.
                    if let Some(ModuleDefId::ModuleId(mod_id)) = item.as_module_def_id() {
                        worklist.push((mod_id, mk_path()));
                    }
                }
            }
        }

        Arc::new(Self { map: import_map })
    }

    /// Returns the `ModPath` needed to import/mention `item`, relative to this crate's root.
    pub fn path_of(&self, item: ItemInNs) -> Option<&ModPath> {
        self.map.get(&item)
    }
}

impl fmt::Debug for ImportMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut importable_paths: Vec<_> = self
            .map
            .iter()
            .map(|(item, modpath)| {
                let ns = match item {
                    ItemInNs::Types(_) => "t",
                    ItemInNs::Values(_) => "v",
                    ItemInNs::Macros(_) => "m",
                };
                format!("- {} ({})", modpath, ns)
            })
            .collect();

        importable_paths.sort();
        f.write_str(&importable_paths.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_db::TestDB;
    use insta::assert_snapshot;
    use ra_db::fixture::WithFixture;
    use ra_db::SourceDatabase;

    fn import_map(ra_fixture: &str) -> String {
        let db = TestDB::with_files(ra_fixture);
        let crate_graph = db.crate_graph();

        let import_maps: Vec<_> = crate_graph
            .iter()
            .filter_map(|krate| {
                let cdata = &crate_graph[krate];
                let name = cdata.display_name.as_ref()?;

                let map = db.import_map(krate);

                Some(format!("{}:\n{:?}", name, map))
            })
            .collect();

        import_maps.join("\n")
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
}
