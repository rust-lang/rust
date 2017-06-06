use semcheck::changes::{Addition, Removal, Change, ChangeSet};

use rustc::hir::def::Export;
use rustc::hir::def::Def::Mod;
use rustc::hir::def_id::DefId;
use rustc::middle::cstore::CrateStore;
use rustc::ty::Visibility::Public;

use std::collections::{HashMap, HashSet, VecDeque};

pub fn traverse(cstore: &CrateStore, new: DefId, old: DefId) -> ChangeSet {
    let mut changes = ChangeSet::default();
    let mut visited = HashSet::new();
    let mut children = HashMap::new();
    let mut mod_queue = VecDeque::new();

    mod_queue.push_back((new, old));

    while let Some((new_did, old_did)) = mod_queue.pop_front() {
        let mut c_new = cstore.item_children(new_did);
        let mut c_old = cstore.item_children(old_did);

        for child in c_new.drain(..).filter(|c| cstore.visibility(c.def.def_id()) == Public) {
            let child_name = String::from(&*child.ident.name.as_str());
            children.entry(child_name).or_insert((None, None)).0 = Some(child);
        }

        for child in c_old.drain(..).filter(|c| cstore.visibility(c.def.def_id()) == Public) {
            let child_name = String::from(&*child.ident.name.as_str());
            children.entry(child_name).or_insert((None, None)).1 = Some(child);
        }

        for (_, items) in children.drain() {
            match items {
                (Some(Export { def: Mod(n), .. }), Some(Export { def: Mod(o), .. })) => {
                    if !visited.insert((n, o)) {
                        mod_queue.push_back((n, o));
                    }
                    changes.register_change(Mod(n), Mod(o));
                },
                (Some(Export { def: n, ..}), Some(Export { def: o, .. })) => {
                    changes.register_change(n, o);
                },
                (Some(new), None) => {
                    changes.add_change(Change::new(Addition, new));
                },
                (None, Some(old)) => {
                    changes.add_change(Change::new(Removal, old));
                },
                (None, None) => unreachable!(),
            }
        }
    }

    changes
}

/*
/// The map of all exports from a crate.
///
/// Mapping paths to item exports.
#[derive(Debug)]
pub struct ExportMap {
    /// The map of paths and item exports.
    paths: PathMap,
    // submods: HashMap<DefId, Vec<Export>>,
    // items: HashMap<DefId, Export>,
}

// TODO: test that we fetch all modules from a crate store (by iterating over all DefIds) it
// defines and comparing them to the set of known DefIds here;

impl ExportMap {
    /// Construct a new export map from a root module's `DefId` and given a `CrateStore`.
    ///
    /// Traverses the descendents of the given module and avoids mutually recursive modules.
    pub fn new(root: DefId, cstore: &CrateStore) -> ExportMap {
        let mut visited = HashSet::new();
        let mut paths = HashMap::new();
        let mut mod_queue = VecDeque::new();
        mod_queue.push_back((root, ExportPath::default()));

        while let Some((mod_id, mod_path)) = mod_queue.pop_front() {
            let mut children = cstore.item_children(mod_id);

            for child in children
                    .drain(..)
                    .filter(|c| cstore.visibility(c.def.def_id()) == Public) {
                let child_name = String::from(&*child.ident.name.as_str());
                match child.def {
                    Def::Mod(submod_id) if !visited.contains(&submod_id) => {
                        visited.insert(submod_id);
                        mod_queue.push_back((submod_id, mod_path.extend(child_name.clone())));
                        paths.insert(mod_path.extend(child_name), child);
                    }
                    _ => {
                        paths.insert(mod_path.extend(child_name), child);
                    }
                }
            }
        }

        ExportMap {
            paths: paths,
        }
    }

    /// Construct an empty export map without filling it.
    ///
    /// This is used for testing and similar tasks.
    #[cfg(test)]
    fn construct(paths: PathMap) -> ExportMap {
        ExportMap {
            paths: paths,
        }
    }

    /// Get a path's corresponding item export, if present.
    pub fn lookup_path(&self, path: &ExportPath) -> Option<&Export> {
        self.paths.get(path)
    }

    /// Compare two change sets, where the current one serves as reference.
    ///
    /// Record the changes determined that way in a `ChangeSet`.
    pub fn compare(&self, other: &ExportMap, from: Checking, changes: &mut ChangeSet) {
        for (path, export) in self.paths
                .iter()
                .filter(|&(p, _)| other.lookup_path(p).is_none()) {
            let change_type = match from {
                Checking::FromNew => Addition,
                Checking::FromOld => Removal,
            };

            changes.add_change(Change::new(change_type, path.clone(), *export));
        }
    }
}
*/

#[cfg(test)]
pub mod tests {
    pub use super::*;

    /* use semcheck::changes::tests as changes;

    use syntax_pos::hygiene::SyntaxContext;
    use syntax_pos::symbol::{Ident, Interner};

    quickcheck! {
        /// If we compare an export map to itself, it shouldn't detect any changes.
        ///
        /// FIXME: this is *very slow*
        fn self_compare_unchanged(change_data: ChangeType) -> bool {
            let mut change_set = ChangeSet::default();

            let export_map = build_export_map(change_data);

            export_map.compare(&export_map, Checking::FromOld, &mut change_set);
            export_map.compare(&export_map, Checking::FromNew, &mut change_set);

            change_set.is_empty()
        }
    } */
}
