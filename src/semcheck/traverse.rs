use semcheck::changes::{Change, ChangeSet};

use rustc::hir::def::Export;
use rustc::hir::def::Def::Mod;
use rustc::hir::def_id::DefId;
use rustc::middle::cstore::CrateStore;
use rustc::ty::Visibility::Public;

use std::collections::{HashMap, HashSet, VecDeque};

/// Traverse the two root modules in an interleaved manner.
///
/// Match up pairs of modules from the two crate versions and compare for changes.
/// Matching children get processed in the same fashion.
pub fn traverse(cstore: &CrateStore, new: DefId, old: DefId) -> ChangeSet {
    let mut changes = ChangeSet::default();
    let mut visited = HashSet::new();
    let mut children = HashMap::new();
    let mut mod_queue = VecDeque::new();

    mod_queue.push_back((new, old));

    while let Some((new_did, old_did)) = mod_queue.pop_front() {
        let mut c_new = cstore.item_children(new_did);
        let mut c_old = cstore.item_children(old_did);

        for child in c_new
                .drain(..)
                .filter(|c| cstore.visibility(c.def.def_id()) == Public) {
            let child_name = String::from(&*child.ident.name.as_str());
            children.entry(child_name).or_insert((None, None)).0 = Some(child);
        }

        for child in c_old
                .drain(..)
                .filter(|c| cstore.visibility(c.def.def_id()) == Public) {
            let child_name = String::from(&*child.ident.name.as_str());
            children.entry(child_name).or_insert((None, None)).1 = Some(child);
        }

        for (_, items) in children.drain() {
            match items {
                (Some(Export { def: Mod(n), .. }), Some(Export { def: Mod(o), .. })) => {
                    if !visited.insert((n, o)) {
                        mod_queue.push_back((n, o));
                    }
                }
                (Some(Export { def: n, .. }), Some(Export { def: o, .. })) => {
                    // changes.add_change(Change::construct(n, o));
                }
                (Some(new), None) => {
                    changes.add_change(Change::new_addition(new));
                }
                (None, Some(old)) => {
                    changes.add_change(Change::new_removal(old));
                }
                (None, None) => unreachable!(),
            }
        }
    }

    changes
}
