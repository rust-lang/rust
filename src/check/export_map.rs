use check::path::{Path, PathMap};

use rustc::hir::def::Def;
use rustc::hir::def_id::DefId;
use rustc::middle::cstore::CrateStore;
use rustc::ty::Visibility::Public;

use std::collections::{HashMap, HashSet, VecDeque};

pub type VisitedModSet = HashSet<DefId>;

pub type ItemSet = HashSet<DefId>;

pub type ModMap = HashMap<DefId, Vec<DefId>>;

pub enum Checking {
    FromOld,
    FromNew,
}

#[derive(Debug)]
pub struct ExportMap {
    visited: VisitedModSet,
    items: ItemSet,
    exports: ModMap,
    paths: PathMap,
    root: DefId,
}

impl ExportMap {
    pub fn new(root: DefId, cstore: &CrateStore) -> ExportMap {
        let mut visited = HashSet::new();
        let mut items = HashSet::new();
        let mut exports = HashMap::new();
        let mut paths = HashMap::new();
        let mut mod_queue = VecDeque::new();
        mod_queue.push_back((root, Path::default()));

        while let Some((mod_id, mod_path)) = mod_queue.pop_front() {
            let mut children = cstore.item_children(mod_id);
            let mut current_children = Vec::new();

            for child in
                children
                    .drain(..)
                    .filter(|c| cstore.visibility(c.def.def_id()) == Public) {
                let child_name = String::from(&*child.ident.name.as_str());
                match child.def {
                    Def::Mod(submod_id) =>
                        if !visited.contains(&submod_id) {
                            visited.insert(submod_id);
                            current_children.push(submod_id);
                            mod_queue.push_back((submod_id, mod_path.extend(child_name)));
                        } else {
                            current_children.push(submod_id);
                        },
                    def => {
                        let def_id = def.def_id();
                        items.insert(def_id);
                        paths.insert(mod_path.extend(child_name), def_id);
                        current_children.push(def_id);
                    },
                }
            }

            exports.insert(mod_id, current_children);
        }

        ExportMap {
            visited: visited,
            items: items,
            exports: exports,
            paths: paths,
            root: root,
        }
    }

    pub fn lookup_path(&self, path: &Path) -> Option<&DefId> {
        self.paths.get(path)
    }

    pub fn compare(&self, other: &ExportMap, from: Checking) {
        for path in self.paths.keys() {
            if other.lookup_path(path).is_none() {
                println!("path differs: {}", path.inner());
            } else {
                println!("path same: {}", path.inner());
            }
        }
    }
}
