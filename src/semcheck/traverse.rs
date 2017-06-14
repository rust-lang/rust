use semcheck::changes::BinaryChangeType;
use semcheck::changes::BinaryChangeType::*;
use semcheck::changes::{Change, ChangeSet};

use rustc::hir::def::Def::*;
use rustc::hir::def::Export;
use rustc::hir::def_id::DefId;
use rustc::ty::TyCtxt;
use rustc::ty::Visibility::Public;

use std::collections::{HashMap, HashSet, VecDeque};

/// Traverse the two root modules in an interleaved manner.
///
/// Match up pairs of modules from the two crate versions and compare for changes.
/// Matching children get processed in the same fashion.
pub fn traverse_modules(tcx: &TyCtxt, old: DefId, new: DefId) -> ChangeSet {
    let cstore = &tcx.sess.cstore;
    let mut changes = ChangeSet::default();
    let mut visited = HashSet::new();
    let mut children = HashMap::new();
    let mut mod_queue = VecDeque::new();

    mod_queue.push_back((old, new));

    while let Some((old_did, new_did)) = mod_queue.pop_front() {
        let mut c_old = cstore.item_children(old_did, tcx.sess);
        let mut c_new = cstore.item_children(new_did, tcx.sess);

        for child in c_old
                .drain(..)
                .filter(|c| cstore.visibility(c.def.def_id()) == Public) {
            let child_name = String::from(&*child.ident.name.as_str());
            children.entry(child_name).or_insert((None, None)).0 = Some(child);
        }

        for child in c_new
                .drain(..)
                .filter(|c| cstore.visibility(c.def.def_id()) == Public) {
            let child_name = String::from(&*child.ident.name.as_str());
            children.entry(child_name).or_insert((None, None)).1 = Some(child);
        }

        for (_, items) in children.drain() {
            match items {
                (Some(Export { def: Mod(o), .. }), Some(Export { def: Mod(n), .. })) => {
                    if !visited.insert((o, n)) {
                        mod_queue.push_back((o, n));
                    }
                }
                (Some(o), Some(n)) => {
                    diff_items(&mut changes, tcx, o, n);
                }
                (Some(old), None) => {
                    changes.add_change(Change::new_removal(old));
                }
                (None, Some(new)) => {
                    changes.add_change(Change::new_addition(new));
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
fn diff_items(changes: &mut ChangeSet, tcx: &TyCtxt, old: Export, new: Export) {
    let mut generics_changes = diff_generics(tcx, old.def.def_id(), new.def.def_id());
    for change_type in generics_changes.drain(..) {
        changes.add_change(Change::new_binary(change_type, old, new));
    }

    let change = match (old.def, new.def) {
        (Struct(_), Struct(_)) => Some(Change::new_binary(Unknown, old, new)),
        (Union(_), Union(_)) => Some(Change::new_binary(Unknown, old, new)),
        (Enum(_), Enum(_)) => Some(Change::new_binary(Unknown, old, new)),
        (Trait(_), Trait(_)) => Some(Change::new_binary(Unknown, old, new)),
        (TyAlias(o), TyAlias(n)) =>
            diff_tyaliases(tcx, o, n).map(|t| Change::new_binary(t, old, new)),
        (Fn(_), Fn(_)) => Some(Change::new_binary(Unknown, old, new)),
        (Const(_), Const(_)) => Some(Change::new_binary(Unknown, old, new)),
        (Static(_, _), Static(_, _)) => Some(Change::new_binary(Unknown, old, new)),
        (Method(_), Method(_)) => Some(Change::new_binary(Unknown, old, new)),
        (Macro(_, _), Macro(_, _)) => Some(Change::new_binary(Unknown, old, new)),
        _ => Some(Change::new_binary(KindDifference, old, new)),
    };

    if let Some(c) = change {
        changes.add_change(c);
    }
}

fn diff_generics(tcx: &TyCtxt, old: DefId, new: DefId) -> Vec<BinaryChangeType> {
    use std::cmp::max;

    let mut ret = Vec::new();

    let old_gen = tcx.generics_of(old);
    let new_gen = tcx.generics_of(new);

    for i in 0..max(old_gen.regions.len(), new_gen.regions.len()) {
        match (old_gen.regions.get(i), new_gen.regions.get(i)) {
            (Some(_ /* old_region */), Some(_ /* new_region */)) => {
                // TODO: handle name changes?
            },
            (Some(_ /* old_region */), None) => {
                ret.push(RegionParameterRemoved);
            },
            (None, Some(_ /* new_region */)) => {
                ret.push(RegionParameterAdded);
            },
            (None, None) => unreachable!(),
        }
    }

    for i in 0..max(old_gen.types.len(), new_gen.types.len()) {
        match (old_gen.types.get(i), new_gen.types.get(i)) {
            (Some(old_type), Some(new_type)) => {
                if old_type.has_default && !new_type.has_default {
                    // TODO: major for sure
                } else if !old_type.has_default && new_type.has_default {
                    // TODO: minor, I guess?
                }
            },
            (Some(old_type), None) => {
                ret.push(TypeParameterRemoved { defaulted: old_type.has_default });
            },
            (None, Some(new_type)) => {
                ret.push(TypeParameterAdded { defaulted: new_type.has_default });
            },
            (None, None) => unreachable!(),
        }
    }

    ret
}

/// Given two type aliases' definitions, compare their types.
fn diff_tyaliases(tcx: &TyCtxt, old: DefId, new: DefId) -> Option<BinaryChangeType> {
    // let cstore = &tcx.sess.cstore;
    /* println!("matching TyAlias'es found");
    println!("old: {:?}", tcx.type_of(old));
    println!("new: {:?}", tcx.type_of(new));*/
    None
}
