use semcheck::changes::BinaryChangeType;
use semcheck::changes::BinaryChangeType::*;
use semcheck::changes::{Change, ChangeCategory, ChangeSet};

use rustc::hir::def::Def::*;
use rustc::hir::def::Export;
use rustc::hir::def_id::DefId;
use rustc::ty::TyCtxt;
use rustc::ty::fold::TypeFolder;
use rustc::ty::subst::{Subst, Substs};
use rustc::ty::Visibility::Public;

use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Default)]
struct IdMapping {
    pub mapping: HashMap<DefId, (DefId, Export, Export)>,
}

impl IdMapping {
    pub fn add(&mut self, old: Export, new: Export) {
        self.mapping.insert(old.def.def_id(), (new.def.def_id(), old, new));
    }

    pub fn get_new_id(&self, old: DefId) -> DefId {
        self.mapping[&old].0
    }
}

/// Traverse the two root modules in an interleaved manner.
///
/// Match up pairs of modules from the two crate versions and compare for changes.
/// Matching children get processed in the same fashion.
pub fn traverse_modules(tcx: &TyCtxt, old: DefId, new: DefId) -> ChangeSet {
    let cstore = &tcx.sess.cstore;
    let mut changes = ChangeSet::default();
    let mut id_mapping = IdMapping::default();
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
                    id_mapping.add(o, n);
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

    for &(_, old, new) in id_mapping.mapping.values() {
        diff_items(&mut changes, &id_mapping, tcx, old, new);
    }

    changes
}

/// Given two items, dispatch to further checks.
///
/// If the two items can't be meaningfully compared because they are of different kinds,
/// we return that difference directly.
fn diff_items(changes: &mut ChangeSet,
              id_mapping: &IdMapping,
              tcx: &TyCtxt,
              old: Export,
              new: Export) {
    let mut check_type = true;
    let mut generics_changes = diff_generics(tcx, old.def.def_id(), new.def.def_id());
    for change_type in generics_changes.drain(..) {
        if ChangeCategory::from(&change_type) == ChangeCategory::Breaking {
            check_type = false;
        }

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
        _ => {
            check_type = false;
            Some(Change::new_binary(KindDifference, old, new))
        },
    };

    if let Some(c) = change {
        changes.add_change(c);
    }

    if check_type {
        let _ = diff_type(id_mapping, tcx, old.def.def_id(), new.def.def_id());
    }
}

struct ComparisonFolder<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for ComparisonFolder<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.tcx }
}

fn diff_type(_: &IdMapping, tcx: &TyCtxt, old: DefId, new: DefId)
    -> Vec<BinaryChangeType>
{
    let res = Vec::new();

    let new_ty = tcx.type_of(new);
    let old_ty = tcx.type_of(old).subst(*tcx, Substs::identity_for_item(*tcx, new));

    println!("old_ty: {:?}", old_ty);
    println!("new_ty: {:?}", new_ty);

    res
}

fn diff_generics(tcx: &TyCtxt, old: DefId, new: DefId) -> Vec<BinaryChangeType> {
    use std::cmp::max;

    let mut ret = Vec::new();

    let old_gen = tcx.generics_of(old);
    let new_gen = tcx.generics_of(new);

    for i in 0..max(old_gen.regions.len(), new_gen.regions.len()) {
        match (old_gen.regions.get(i), new_gen.regions.get(i)) {
            (Some(_ /* old_region */), None) => {
                ret.push(RegionParameterRemoved);
            },
            (None, Some(_ /* new_region */)) => {
                ret.push(RegionParameterAdded);
            },
            (Some(_), Some(_)) => (),
            (None, None) => unreachable!(),
        }
    }

    for i in 0..max(old_gen.types.len(), new_gen.types.len()) {
        match (old_gen.types.get(i), new_gen.types.get(i)) {
            (Some(old_type), Some(new_type)) => {
                if old_type.has_default && !new_type.has_default {
                    // TODO: major for sure
                    ret.push(TypeParameterRemoved { defaulted: true });
                    ret.push(TypeParameterAdded { defaulted: false });
                } else if !old_type.has_default && new_type.has_default {
                    // TODO: minor, I guess?
                    ret.push(TypeParameterRemoved { defaulted: false });
                    ret.push(TypeParameterAdded { defaulted: true });
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
fn diff_tyaliases(/*tcx*/ _: &TyCtxt, /*old*/ _: DefId, /*new*/ _: DefId)
    -> Option<BinaryChangeType> {
    // let cstore = &tcx.sess.cstore;
    /* println!("matching TyAlias'es found");
    println!("old: {:?}", tcx.type_of(old));
    println!("new: {:?}", tcx.type_of(new));*/
    None
}
