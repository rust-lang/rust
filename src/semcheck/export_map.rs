use semcheck::changes::{Addition, Removal, Change, ChangeSet};
use semcheck::path::{ExportPath, PathMap};

use rustc::hir::def::{Def, Export};
use rustc::hir::def_id::DefId;
use rustc::middle::cstore::CrateStore;
use rustc::ty::Visibility::Public;

use std::collections::{HashMap, HashSet, VecDeque};

/// A marker to identify the current export map.
pub enum Checking {
    /// We're running from the old crate's export map.
    FromOld,
    /// We're running from the new crate's export map.
    FromNew,
}

/// The map of all exports from a crate.
///
/// Mapping paths to item exports.
#[derive(Debug)]
pub struct ExportMap {
    /// The map of paths and item exports.
    paths: PathMap,
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

    // TODO: test that the compairson with oneself doesn't modify the change set
}
