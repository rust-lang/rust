use semcheck::changes::BinaryChangeType::*;
use semcheck::changes::{Change, ChangeSet};

use rustc::hir::def::Def::*;
use rustc::hir::def::Export;
use rustc::hir::def_id::DefId;
use rustc::middle::cstore::CrateStore;
use rustc::ty::TyCtxt;
use rustc::ty::Visibility::Public;

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet, VecDeque};

/// Traverse the two root modules in an interleaved manner.
///
/// Match up pairs of modules from the two crate versions and compare for changes.
/// Matching children get processed in the same fashion.
pub fn traverse_modules(tcx: &TyCtxt, new: DefId, old: DefId) -> ChangeSet {
    let cstore = &tcx.sess.cstore;
    let mut changes = ChangeSet::default();
    let mut visited = HashSet::new();
    let mut children = HashMap::new();
    let mut mod_queue = VecDeque::new();

    mod_queue.push_back((new, old));

    while let Some((new_did, old_did)) = mod_queue.pop_front() {
        let mut c_new = cstore.item_children(new_did, tcx.sess);
        let mut c_old = cstore.item_children(old_did, tcx.sess);

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
                (Some(n), Some(o)) => {
                    if let Some(change) = diff_items(cstore.borrow(), n, o) {
                        changes.add_change(change);
                    }
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

/// Given two items, dispatch to further checks.
///
/// If the two items can't be meaningfully compared because they are of different kinds,
/// we return that difference directly.
pub fn diff_items(_: &CrateStore, new: Export, old: Export) -> Option<Change> {
    match (new.def, old.def) {
        (Struct(_), Struct(_)) => Some(Change::new_binary(Unknown, old, new)),
        (Union(_), Union(_)) => Some(Change::new_binary(Unknown, old, new)),
        (Enum(_), Enum(_)) => Some(Change::new_binary(Unknown, old, new)),
        (Trait(_), Trait(_)) => Some(Change::new_binary(Unknown, old, new)),
        (TyAlias(_), TyAlias(_)) => Some(Change::new_binary(Unknown, old, new)),
        (Fn(_), Fn(_)) => Some(Change::new_binary(Unknown, old, new)),
        (Const(_), Const(_)) => Some(Change::new_binary(Unknown, old, new)),
        (Static(_, _), Static(_, _)) => Some(Change::new_binary(Unknown, old, new)),
        (Method(_), Method(_)) => Some(Change::new_binary(Unknown, old, new)),
        (Macro(_, _), Macro(_, _)) => Some(Change::new_binary(Unknown, old, new)),
        _ => Some(Change::new_binary(KindDifference, old, new)),
    }
}
