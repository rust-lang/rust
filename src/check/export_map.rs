use check::change::{Addition, Removal, Change, ChangeSet};
use check::path::{Path, PathMap};

use rustc::hir::def::{Def, Export};
use rustc::hir::def_id::DefId;
use rustc::middle::cstore::CrateStore;
use rustc::ty::Visibility::Public;

use std::collections::{HashMap, HashSet, VecDeque};

pub type VisitedModSet = HashSet<DefId>;

pub type ModMap = HashMap<DefId, Vec<DefId>>;

pub enum Checking {
    FromOld,
    FromNew,
}

#[derive(Debug)]
pub struct ExportMap {
    visited: VisitedModSet,
    paths: PathMap,
}

impl ExportMap {
    pub fn new(root: DefId, cstore: &CrateStore) -> ExportMap {
        let mut visited = HashSet::new();
        let mut paths = HashMap::new();
        let mut mod_queue = VecDeque::new();
        mod_queue.push_back((root, Path::default()));

        while let Some((mod_id, mod_path)) = mod_queue.pop_front() {
            let mut children = cstore.item_children(mod_id);

            for child in
                children
                    .drain(..)
                    .filter(|c| cstore.visibility(c.def.def_id()) == Public) {
                let child_name = String::from(&*child.ident.name.as_str());
                match child.def {
                    Def::Mod(submod_id) if !visited.contains(&submod_id) => {
                        visited.insert(submod_id);
                        mod_queue.push_back((submod_id, mod_path.extend(child_name.clone())));
                        paths.insert(mod_path.extend(child_name), child);
                    },
                    _ => {
                        paths.insert(mod_path.extend(child_name), child);
                    },
                }
            }
        }

        ExportMap {
            visited: visited,
            paths: paths,
        }
    }

    pub fn lookup_path(&self, path: &Path) -> Option<&Export> {
        self.paths.get(path)
    }

    pub fn compare(&self, other: &ExportMap, from: Checking, changes: &mut ChangeSet) {
        for (path, export) in
            self.paths
                .iter()
                .filter(|&(p, _)| other.lookup_path(p).is_none()) {
            let change_type = match from {
                Checking::FromNew => Addition,
                Checking::FromOld => Removal,
            };

            changes.add_change(Change::new(change_type, path.clone(), export.clone()));
        }
    }
}
