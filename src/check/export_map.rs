use rustc::hir::def::Def;
use rustc::hir::def_id::*;
use rustc::middle::cstore::CrateStore;
use rustc::ty::Visibility::Public;

use std::collections::{HashMap, HashSet, VecDeque};

pub type VisitedModSet = HashSet<DefId>;

pub type ItemSet = HashSet<DefId>;

pub type ModMap = HashMap<DefId, Vec<DefId>>;

#[derive(Debug)]
pub struct ExportMap {
    visited_mods: VisitedModSet,
    items: ItemSet,
    exports: ModMap,
    root: DefId,
}

impl ExportMap {
    pub fn new(root: DefId, cstore: &CrateStore) -> ExportMap {
        let (visited, items, exports) = walk_mod(root, cstore);
        ExportMap {
            visited_mods: visited,
            items: items,
            exports: exports,
            root: root,
        }
    }
}

fn walk_mod(root: DefId, cstore: &CrateStore) -> (VisitedModSet, ItemSet, ModMap) {
    let mut visited = HashSet::new();
    let mut items = HashSet::new();
    let mut exports = HashMap::new();

    let mut mod_queue = VecDeque::new();
    mod_queue.push_back(root);

    while let Some(mod_id) = mod_queue.pop_front() {
        let mut children = cstore.item_children(mod_id);
        let mut current_children = Vec::new();

        for child in children.drain(..).filter(|c| cstore.visibility(c.def.def_id()) == Public) {
            match child.def {
                Def::Mod(submod_id) =>
                    if !visited.contains(&submod_id) {
                        visited.insert(submod_id);
                        current_children.push(submod_id);
                        mod_queue.push_back(submod_id);
                    } else {
                        current_children.push(submod_id);
                    },
                def => {
                    let def_id = def.def_id();
                    items.insert(def_id);
                    current_children.push(def_id);
                },
            }
        }

        exports.insert(mod_id, current_children);
    }

    (visited, items, exports)
}
