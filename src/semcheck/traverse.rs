//! The traversal logic collecting changes in between crate versions.
//!
//! The changes get collected in multiple passes, and recorded in a `ChangeSet`.
//! The initial pass matches items by name in the module hierarchy, registering item removal
//! and addition, as well as structural changes to ADTs, type- or region parameters, and
//! function signatures. The second pass then proceeds find non-public items that are named
//! differently, yet are compatible in their usage. The (currently not implemented) third pass
//! performs the same analysis on trait bounds. The fourth and final pass now uses the
//! information collected in the previous passes to compare the types of all item pairs having
//! been matched.

use rustc::hir::def::{CtorKind, Def};
use rustc::hir::def_id::DefId;
use rustc::infer::InferCtxt;
use rustc::traits::{FulfillmentContext, FulfillmentError, Obligation, ObligationCause};
use rustc::ty::{AssociatedItem, ParamEnv, TraitRef, Ty, TyCtxt};
use rustc::ty::Visibility::Public;
use rustc::ty::fold::TypeFoldable;
use rustc::ty::subst::{Subst, Substs};

use semcheck::changes::ChangeType::*;
use semcheck::changes::ChangeSet;
use semcheck::mapping::{IdMapping, InherentEntry, InherentImplSet, NameMapping};
use semcheck::mismatch::Mismatch;
use semcheck::translate::{BottomUpRegionFolder, TranslationContext};

use std::collections::{BTreeMap, HashSet, VecDeque};

use syntax::symbol::Symbol;

/// The main entry point to our analysis passes.
///
/// Set up the necessary data structures and run the analysis passes.
pub fn run_analysis<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, old: DefId, new: DefId)
    -> ChangeSet<'tcx>
{
    let mut changes = Default::default();
    let mut id_mapping = IdMapping::new(old.krate, new.krate);

    // first pass
    diff_structure(&mut changes, &mut id_mapping, tcx, old, new);

    // second pass
    {
        let mut mismatch = Mismatch::new(tcx, &mut id_mapping);
        mismatch.process();
    }

    // third pass
    for (old, new) in id_mapping.items() {
        diff_bounds(&mut changes, &id_mapping, tcx, old, new);
    }

    // fourth pass
    for (old, new) in id_mapping.items() {
        diff_types(&mut changes, &id_mapping, tcx, old, new);
    }

    // fourth pass still
    diff_inherent_impls(&mut changes, &id_mapping, tcx);
    diff_trait_impls(&mut changes, &id_mapping, tcx);

    changes
}

// Below functions constitute the first pass of analysis, in which module structure, ADT
// structure, public and private status of items, and generics are examined for changes.

/// Given two crate root modules, compare their exports and their structure.
///
/// Traverse the two root modules in an interleaved manner, matching up pairs of modules
/// from the two crate versions and compare for changes. Matching children get processed
/// in the same fashion.
// TODO: clean up and simplify.
fn diff_structure<'a, 'tcx>(changes: &mut ChangeSet,
                            id_mapping: &mut IdMapping,
                            tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            old: DefId,
                            new: DefId) {
    use rustc::hir::def::Def::*;

    let cstore = &tcx.sess.cstore;
    let mut visited = HashSet::new();
    let mut children = NameMapping::default();
    let mut mod_queue = VecDeque::new();
    // Additions and removals are processed with a delay to avoid creating multiple path change
    // entries. This is necessary, since the order in which added or removed paths are found wrt
    // each other and their item's definition can't be relied upon.
    let mut removals = Vec::new();
    let mut additions = Vec::new();

    mod_queue.push_back((old, new, Public, Public));

    while let Some((old_def_id, new_def_id, old_vis, new_vis)) = mod_queue.pop_front() {
        children.add(cstore.item_children(old_def_id, tcx.sess),
                     cstore.item_children(new_def_id, tcx.sess));

        for items in children.drain() {
            match items {
                (Some(o), Some(n)) => {
                    if let (Mod(o_def_id), Mod(n_def_id)) = (o.def, n.def) {
                        if visited.insert((o_def_id, n_def_id)) {
                            let o_vis = if old_vis == Public {
                                cstore.visibility(o_def_id)
                            } else {
                                old_vis
                            };
                            let n_vis = if new_vis == Public {
                                cstore.visibility(n_def_id)
                            } else {
                                new_vis
                            };

                            if o_vis != n_vis {
                                changes.new_change(o_def_id,
                                                   n_def_id,
                                                   o.ident.name,
                                                   tcx.def_span(o_def_id),
                                                   tcx.def_span(n_def_id),
                                                   true);

                                if o_vis == Public && n_vis != Public {
                                    changes.add_change(ItemMadePrivate, o_def_id, None);
                                } else if o_vis != Public && n_vis == Public {
                                    changes.add_change(ItemMadePublic, o_def_id, None);
                                }
                            }

                            mod_queue.push_back((o_def_id, n_def_id, o_vis, n_vis));
                        }
                    } else if id_mapping.add_export(o.def, n.def) {
                        // struct constructors are weird/hard - let's go shopping!
                        if let (StructCtor(_, _), StructCtor(_, _)) = (o.def, n.def) {
                            continue;
                        }

                        let o_def_id = o.def.def_id();
                        let n_def_id = n.def.def_id();
                        let o_vis = if old_vis == Public {
                            cstore.visibility(o_def_id)
                        } else {
                            old_vis
                        };
                        let n_vis = if new_vis == Public {
                            cstore.visibility(n_def_id)
                        } else {
                            new_vis
                        };

                        let output = o_vis == Public || n_vis == Public;
                        changes.new_change(o_def_id,
                                           n_def_id,
                                           o.ident.name,
                                           tcx.def_span(o_def_id),
                                           tcx.def_span(n_def_id),
                                           output);

                        if o_vis == Public && n_vis != Public {
                            changes.add_change(ItemMadePrivate, o_def_id, None);
                        } else if o_vis != Public && n_vis == Public {
                            changes.add_change(ItemMadePublic, o_def_id, None);
                        }

                        match (o.def, n.def) {
                            // (matching) things we don't care about (for now)
                            (Mod(_), Mod(_)) |
                            (AssociatedTy(_), AssociatedTy(_)) |
                            (PrimTy(_), PrimTy(_)) |
                            (TyParam(_), TyParam(_)) |
                            (SelfTy(_, _), SelfTy(_, _)) |
                            (StructCtor(_, _), StructCtor(_, _)) |
                            (VariantCtor(_, _), VariantCtor(_, _)) |
                            (AssociatedConst(_), AssociatedConst(_)) |
                            (Local(_), Local(_)) |
                            (Upvar(_, _, _), Upvar(_, _, _)) |
                            (Label(_), Label(_)) |
                            (GlobalAsm(_), GlobalAsm(_)) |
                            (Macro(_, _), Macro(_, _)) |
                            (Variant(_), Variant(_)) |
                            (Const(_), Const(_)) |
                            (Static(_, _), Static(_, _)) |
                            (Err, Err) => {},
                            (Fn(_), Fn(_)) |
                            (Method(_), Method(_)) => {
                                diff_generics(changes,
                                              id_mapping,
                                              tcx,
                                              true,
                                              o_def_id,
                                              n_def_id);
                                diff_fn(changes, tcx, o.def, n.def);
                            },
                            (TyAlias(_), TyAlias(_)) => {
                                diff_generics(changes,
                                              id_mapping,
                                              tcx,
                                              false,
                                              o_def_id,
                                              n_def_id);
                            },
                            (Struct(_), Struct(_)) |
                            (Union(_), Union(_)) |
                            (Enum(_), Enum(_)) => {
                                diff_generics(changes,
                                              id_mapping,
                                              tcx,
                                              false,
                                              o_def_id,
                                              n_def_id);
                                diff_adts(changes, id_mapping, tcx, o.def, n.def);
                            },
                            (Trait(_), Trait(_)) => {
                                diff_generics(changes,
                                              id_mapping,
                                              tcx,
                                              false,
                                              o_def_id,
                                              n_def_id);
                                diff_traits(changes,
                                            id_mapping,
                                            tcx,
                                            o_def_id,
                                            n_def_id,
                                            output);
                            },
                            // non-matching item pair - register the difference and abort
                            _ => {
                                changes.add_change(KindDifference, o_def_id, None);
                            },
                        }
                    }
                }
                (Some(o), None) => {
                    // struct constructors are weird/hard - let's go shopping!
                    if let StructCtor(_, _) = o.def {
                        continue;
                    }

                    let o_def_id = o.def.def_id();

                    if old_vis == Public && cstore.visibility(o_def_id) == Public {
                        // delay the handling of removals until the id mapping is complete
                        removals.push(o);
                    }
                }
                (None, Some(n)) => {
                    // struct constructors are weird/hard - let's go shopping!
                    if let StructCtor(_, _) = n.def {
                        continue;
                    }

                    let n_def_id = n.def.def_id();

                    if new_vis == Public && cstore.visibility(n_def_id) == Public {
                        // delay the handling of additions until the id mapping is complete
                        additions.push(n);
                    }
                }
                (None, None) => unreachable!(),
            }
        }
    }

    // finally, process item additions and removals
    for n in additions {
        let n_def_id = n.def.def_id();

        if !id_mapping.contains_new_id(n_def_id) {
            id_mapping.add_non_mapped(n_def_id);
        }

        changes.new_path_change(n_def_id, n.ident.name, tcx.def_span(n_def_id));
        changes.add_path_addition(n_def_id, n.span);
    }

    for o in removals {
        let o_def_id = o.def.def_id();

        // reuse an already existing path change entry, if possible
        if id_mapping.contains_old_id(o_def_id) {
            let n_def_id = id_mapping.get_new_id(o_def_id).unwrap();
            changes.new_path_change(n_def_id, o.ident.name, tcx.def_span(n_def_id));
            changes.add_path_removal(n_def_id, o.span);
        } else {
            id_mapping.add_non_mapped(o_def_id);
            changes.new_path_change(o_def_id, o.ident.name, tcx.def_span(o_def_id));
            changes.add_path_removal(o_def_id, o.span);
        }
    }
}

/// Given two fn items, perform structural checks.
fn diff_fn(changes: &mut ChangeSet, tcx: TyCtxt, old: Def, new: Def) {
    let old_def_id = old.def_id();
    let new_def_id = new.def_id();

    let old_const = tcx.is_const_fn(old_def_id);
    let new_const = tcx.is_const_fn(new_def_id);

    if old_const != new_const {
        changes.add_change(FnConstChanged { now_const: new_const }, old_def_id, None);
    }
}

/// Given two method items, perform structural checks.
fn diff_method(changes: &mut ChangeSet, tcx: TyCtxt, old: AssociatedItem, new: AssociatedItem) {
    if old.method_has_self_argument != new.method_has_self_argument {
        changes.add_change(MethodSelfChanged { now_self: new.method_has_self_argument },
                           old.def_id,
                           None);
    }

    diff_fn(changes, tcx, Def::Method(old.def_id), Def::Method(new.def_id));
}

/// Given two ADT items, perform structural checks.
///
/// This establishes the needed correspondence between non-toplevel items such as enum variants,
/// struct and enum fields etc.
fn diff_adts(changes: &mut ChangeSet,
             id_mapping: &mut IdMapping,
             tcx: TyCtxt,
             old: Def,
             new: Def) {
    use rustc::hir::def::Def::*;

    let old_def_id = old.def_id();
    let new_def_id = new.def_id();

    let (old_def, new_def) = match (old, new) {
        (Struct(_), Struct(_)) |
        (Union(_), Union(_)) |
        (Enum(_), Enum(_)) => (tcx.adt_def(old_def_id), tcx.adt_def(new_def_id)),
        _ => return,
    };

    let mut variants = BTreeMap::new();
    let mut fields = BTreeMap::new();

    for variant in &old_def.variants {
        variants.entry(variant.name).or_insert((None, None)).0 = Some(variant);
    }

    for variant in &new_def.variants {
        variants.entry(variant.name).or_insert((None, None)).1 = Some(variant);
    }

    for items in variants.values() {
        match *items {
            (Some(old), Some(new)) => {
                for field in &old.fields {
                    fields.entry(field.name).or_insert((None, None)).0 = Some(field);
                }

                for field in &new.fields {
                    fields.entry(field.name).or_insert((None, None)).1 = Some(field);
                }

                let mut total_private = true;
                let mut total_public = true;

                for items2 in fields.values() {
                    if let Some(o) = items2.0 {
                        let public = o.vis == Public;
                        total_public &= public;
                        total_private &= !public;
                    }
                }

                if old.ctor_kind != new.ctor_kind {
                    let c = VariantStyleChanged {
                        now_struct: new.ctor_kind == CtorKind::Fictive,
                        total_private: total_private,
                    };
                    changes.add_change(c, old_def_id, Some(tcx.def_span(new.did)));

                    continue;
                }

                for items2 in fields.values() {
                    match *items2 {
                        (Some(o), Some(n)) => {
                            id_mapping.add_subitem(old_def_id, o.did, n.did);

                            if o.vis != Public && n.vis == Public {
                                changes.add_change(ItemMadePublic,
                                                   old_def_id,
                                                   Some(tcx.def_span(n.did)));
                            } else if o.vis == Public && n.vis != Public {
                                changes.add_change(ItemMadePrivate,
                                                   old_def_id,
                                                   Some(tcx.def_span(n.did)));
                            }
                        },
                        (Some(o), None) => {
                            let c = VariantFieldRemoved {
                                public: o.vis == Public,
                                total_public: total_public
                            };
                            changes.add_change(c, old_def_id, Some(tcx.def_span(o.did)));
                        },
                        (None, Some(n)) => {
                            let c = VariantFieldAdded {
                                public: n.vis == Public,
                                total_public: total_public
                            };
                            changes.add_change(c, old_def_id, Some(tcx.def_span(n.did)));
                        },
                        (None, None) => unreachable!(),
                    }
                }

                fields.clear();
            },
            (Some(old), None) => {
                changes.add_change(VariantRemoved, old_def_id, Some(tcx.def_span(old.did)));
            },
            (None, Some(new)) => {
                changes.add_change(VariantAdded, old_def_id, Some(tcx.def_span(new.did)));
            },
            (None, None) => unreachable!(),
        }
    }

    for impl_def_id in tcx.inherent_impls(old_def_id).iter() {
        for item_def_id in tcx.associated_item_def_ids(*impl_def_id).iter() {
            let item = tcx.associated_item(*item_def_id);
            id_mapping.add_inherent_item(old_def_id,
                                         item.kind,
                                         item.name,
                                         *impl_def_id,
                                         *item_def_id);
        }
    }

    for impl_def_id in tcx.inherent_impls(new_def_id).iter() {
        for item_def_id in tcx.associated_item_def_ids(*impl_def_id).iter() {
            let item = tcx.associated_item(*item_def_id);
            id_mapping.add_inherent_item(new_def_id,
                                         item.kind,
                                         item.name,
                                         *impl_def_id,
                                         *item_def_id);
        }
    }
}

/// Given two trait items, perform structural checks.
///
/// This establishes the needed correspondence between non-toplevel items found in the trait
/// definition.
fn diff_traits(changes: &mut ChangeSet,
               id_mapping: &mut IdMapping,
               tcx: TyCtxt,
               old: DefId,
               new: DefId,
               output: bool) {
    use rustc::hir::Unsafety::Unsafe;

    let old_unsafety = tcx.trait_def(old).unsafety;
    let new_unsafety = tcx.trait_def(new).unsafety;

    if old_unsafety != new_unsafety {
        let change_type = TraitUnsafetyChanged {
            now_unsafe: new_unsafety == Unsafe,
        };

        changes.add_change(change_type, old, None);
    }

    let mut items = BTreeMap::new();

    for old_def_id in tcx.associated_item_def_ids(old).iter() {
        let item = tcx.associated_item(*old_def_id);
        items.entry(item.name).or_insert((None, None)).0 =
            tcx.describe_def(*old_def_id).map(|d| (d, item));
    }

    for new_def_id in tcx.associated_item_def_ids(new).iter() {
        let item = tcx.associated_item(*new_def_id);
        items.entry(item.name).or_insert((None, None)).1 =
            tcx.describe_def(*new_def_id).map(|d| (d, item));
    }

    for (name, item_pair) in &items {
        match *item_pair {
            (Some((old_def, old_item)), Some((new_def, new_item))) => {
                let old_def_id = old_def.def_id();
                let new_def_id = new_def.def_id();

                id_mapping.add_trait_item(old_def, new_def, old);
                changes.new_change(old_def_id,
                                   new_def_id,
                                   *name,
                                   tcx.def_span(old_def_id),
                                   tcx.def_span(new_def_id),
                                   output);

                diff_generics(changes, id_mapping, tcx, true, old_def_id, new_def_id);
                diff_method(changes, tcx, old_item, new_item);
            },
            (Some((_, old_item)), None) => {
                let change_type = TraitItemRemoved {
                    defaulted: old_item.defaultness.has_value(),
                };
                changes.add_change(change_type, old, Some(tcx.def_span(old_item.def_id)));
                id_mapping.add_non_mapped(old_item.def_id);
            },
            (None, Some((_, new_item))) => {
                let change_type = TraitItemAdded {
                    defaulted: new_item.defaultness.has_value(),
                };
                changes.add_change(change_type, old, Some(tcx.def_span(new_item.def_id)));
                id_mapping.add_non_mapped(new_item.def_id);
            },
            (None, None) => unreachable!(),
        }
    }
}

/// Given two items, compare their type and region parameter sets.
fn diff_generics(changes: &mut ChangeSet,
                 id_mapping: &mut IdMapping,
                 tcx: TyCtxt,
                 is_fn: bool,
                 old: DefId,
                 new: DefId) {
    use std::cmp::max;

    let mut found = Vec::new();

    let old_gen = tcx.generics_of(old);
    let new_gen = tcx.generics_of(new);

    for i in 0..max(old_gen.regions.len(), new_gen.regions.len()) {
        match (old_gen.regions.get(i), new_gen.regions.get(i)) {
            (Some(old_region), Some(new_region)) => {
                id_mapping.add_internal_item(old_region.def_id, new_region.def_id);
            },
            (Some(_ /* old_region */), None) => {
                found.push(RegionParameterRemoved);
            },
            (None, Some(_ /* new_region */)) => {
                found.push(RegionParameterAdded);
            },
            (None, None) => unreachable!(),
        }
    }

    for i in 0..max(old_gen.types.len(), new_gen.types.len()) {
        match (old_gen.types.get(i), new_gen.types.get(i)) {
            (Some(old_type), Some(new_type)) => {
                if old_type.has_default && !new_type.has_default {
                    found.push(TypeParameterRemoved { defaulted: true });
                    found.push(TypeParameterAdded { defaulted: false });
                } else if !old_type.has_default && new_type.has_default {
                    found.push(TypeParameterRemoved { defaulted: false });
                    found.push(TypeParameterAdded { defaulted: true });
                }

                debug!("in item {:?} / {:?}:\n  type param pair: {:?}, {:?}",
                       old, new, old_type, new_type);

                id_mapping.add_internal_item(old_type.def_id, new_type.def_id);
                id_mapping.add_type_param(*old_type);
                id_mapping.add_type_param(*new_type);
            },
            (Some(old_type), None) => {
                found.push(TypeParameterRemoved { defaulted: old_type.has_default });
                id_mapping.add_type_param(*old_type);
                id_mapping.add_non_mapped(old_type.def_id);
            },
            (None, Some(new_type)) => { // FIXME: is_fn could be used in a more elegant fashion
                found.push(TypeParameterAdded { defaulted: new_type.has_default || is_fn });
                id_mapping.add_type_param(*new_type);
                id_mapping.add_non_mapped(new_type.def_id);
            },
            (None, None) => unreachable!(),
        }
    }

    for change_type in found.drain(..) {
        changes.add_change(change_type, old, None);
    }
}

// Below functions constitute the third pass of analysis, in which parameter bounds of matching
// items are compared for changes and used to determine matching relationships between items not
// being exported.

/// Given two items, compare the bounds on their type and region parameters.
fn diff_bounds<'a, 'tcx>(_changes: &mut ChangeSet,
                         _id_mapping: &IdMapping,
                         _tcx: TyCtxt<'a, 'tcx, 'tcx>,
                         _old: Def,
                         _new: Def) {
}

// Below functions constitute the fourth and last pass of analysis, in which the types of
// matching items are compared for changes.

/// Given two items, compare their types.
fn diff_types<'a, 'tcx>(changes: &mut ChangeSet<'tcx>,
                        id_mapping: &IdMapping,
                        tcx: TyCtxt<'a, 'tcx, 'tcx>,
                        old: Def,
                        new: Def) {
    use rustc::hir::def::Def::*;

    let old_def_id = old.def_id();
    let new_def_id = new.def_id();

    if changes.item_breaking(old_def_id) ||
            id_mapping.get_trait_def(&old_def_id)
                .map_or(false, |did| changes.trait_item_breaking(did)) {
        return;
    }

    match old {
        TyAlias(_) => {
            cmp_types(changes,
                      id_mapping,
                      tcx,
                      old_def_id,
                      new_def_id,
                      tcx.type_of(old_def_id),
                      tcx.type_of(new_def_id));
        },
        Fn(_) | Method(_) => {
            let old_fn_sig = tcx.type_of(old_def_id).fn_sig(tcx);
            let new_fn_sig = tcx.type_of(new_def_id).fn_sig(tcx);

            cmp_types(changes,
                      id_mapping,
                      tcx,
                      old_def_id,
                      new_def_id,
                      tcx.mk_fn_ptr(old_fn_sig),
                      tcx.mk_fn_ptr(new_fn_sig));
        },
        Struct(_) | Enum(_) | Union(_) => {
            if let Some(children) = id_mapping.children_of(old_def_id) {
                for (o_def_id, n_def_id) in children {
                    let o_ty = tcx.type_of(o_def_id);
                    let n_ty = tcx.type_of(n_def_id);

                    cmp_types(changes, id_mapping, tcx, old_def_id, new_def_id, o_ty, n_ty);
                }
            }
        },
        _ => (),
    }
}

/// Compare the inherent implementations of items.
fn diff_inherent_impls<'a, 'tcx>(changes: &mut ChangeSet<'tcx>,
                                 id_mapping: &IdMapping,
                                 tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let to_new = TranslationContext::target_new(tcx, id_mapping, false);
    let to_old = TranslationContext::target_old(tcx, id_mapping, false);

    for (orig_item, orig_impls) in id_mapping.inherent_impls() {
        let (trans, err_type) = if id_mapping.in_old_crate(orig_item.parent_def_id) {
            (&to_new, AssociatedItemRemoved)
        } else if id_mapping.in_new_crate(orig_item.parent_def_id) {
            (&to_old, AssociatedItemAdded)
        } else {
            unreachable!()
        };

        for &(orig_impl_def_id, orig_item_def_id) in orig_impls {
            let target_impls = if let Some(impls) = trans
                .translate_inherent_entry(orig_item)
                .and_then(|item| id_mapping.get_inherent_impls(&item))
            {
                impls
            } else {
                continue;
            };

            let match_found = target_impls
                .iter()
                .any(|&(target_impl_def_id, target_item_def_id)| {
                    match_inherent_impl(id_mapping,
                                        tcx,
                                        trans,
                                        orig_impl_def_id,
                                        orig_item_def_id,
                                        target_impl_def_id,
                                        target_item_def_id)
                });

            if !match_found {
                let item_span = tcx.def_span(orig_item_def_id);

                changes.new_change(orig_item_def_id,
                                   orig_item_def_id,
                                   orig_item.name,
                                   item_span,
                                   item_span,
                                   true);
                changes.add_change(err_type.clone(), orig_item_def_id, None);
            }
        }
    }
}

/// Compare the implementations of traits.
fn diff_trait_impls<'a, 'tcx>(changes: &mut ChangeSet<'tcx>,
                              id_mapping: &IdMapping,
                              tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let all_impls = tcx.sess.cstore.implementations_of_trait(None);

    for old_impl_def_id in all_impls.iter().filter(|&did| id_mapping.in_old_crate(*did)) {
        let old_trait_def_id = tcx.impl_trait_ref(*old_impl_def_id).unwrap().def_id;
        if id_mapping.get_new_id(old_trait_def_id).is_none() {
            continue;
        }

        if !match_trait_impl(id_mapping, tcx, *old_impl_def_id) {
            let impl_span = tcx.def_span(*old_impl_def_id);

            changes.new_change(*old_impl_def_id,
                               *old_impl_def_id,
                               Symbol::intern("impl"),
                               impl_span,
                               impl_span,
                               true);
            changes.add_change(TraitImplTightened, *old_impl_def_id, None);
        }
    }

    for new_impl_def_id in all_impls.iter().filter(|&did| id_mapping.in_new_crate(*did)) {
        let new_trait_def_id = tcx.impl_trait_ref(*new_impl_def_id).unwrap().def_id;
        if id_mapping.get_old_id(new_trait_def_id).is_none() {
            continue;
        }

        if !match_trait_impl(id_mapping, tcx, *new_impl_def_id) {
            let impl_span = tcx.def_span(*new_impl_def_id);

            changes.new_change(*new_impl_def_id,
                               *new_impl_def_id,
                               Symbol::intern("impl"),
                               impl_span,
                               impl_span,
                               true);
            changes.add_change(TraitImplLoosened, *new_impl_def_id, None);
        }
    }
}

/// Compare two types and possibly register the error.
fn cmp_types<'a, 'tcx>(changes: &mut ChangeSet<'tcx>,
                       id_mapping: &IdMapping,
                       tcx: TyCtxt<'a, 'tcx, 'tcx>,
                       old_def_id: DefId,
                       new_def_id: DefId,
                       old: Ty<'tcx>,
                       new: Ty<'tcx>) {
    use rustc::infer::InferOk;
    use rustc::middle::free_region::FreeRegionMap;
    use rustc::middle::region::RegionMaps;
    use rustc::ty::{Lift, ReEarlyBound};
    use rustc::ty::TypeVariants::*;
    use syntax_pos::DUMMY_SP;

    info!("comparing types of {:?} / {:?}:\n  {:?} / {:?}", old_def_id, new_def_id, old, new);

    let to_new = TranslationContext::target_new(tcx, id_mapping, false);
    let to_old = TranslationContext::target_old(tcx, id_mapping, false);

    let substs = Substs::identity_for_item(tcx, new_def_id);
    let old = to_new.translate_item_type(old_def_id, old);

    tcx.infer_ctxt().enter(|infcx| {
        let new_substs = if new.is_fn() {
            let has_self = tcx.generics_of(new_def_id).has_self;
            Substs::for_item(infcx.tcx, new_def_id, |def, _| {
                infcx.region_var_for_def(DUMMY_SP, def)
            }, |def, substs| {
                if def.index == 0 && has_self { // `Self` is special
                    tcx.mk_param_from_def(def)
                } else {
                    infcx.type_var_for_def(DUMMY_SP, def, substs)
                }
            })
        } else {
            Substs::for_item(tcx, new_def_id, |def, _| {
                tcx.mk_region(ReEarlyBound(def.to_early_bound_region_data()))
            }, |def, _| if id_mapping.is_non_mapped_defaulted_type_param(&def.def_id) {
                tcx.type_of(def.def_id)
            } else {
                tcx.mk_param_from_def(def)
            })
        };

        let new = new.subst(infcx.tcx, new_substs);
        // let old = old.subst(infcx.tcx, substs);

        let new_param_env = tcx.param_env(new_def_id).subst(infcx.tcx, new_substs);

        let error = infcx
            .at(&ObligationCause::dummy(), new_param_env)
            .eq(old, new)
            .map(|InferOk { obligations: o, .. }| { assert_eq!(o, vec![]); });

        let mut folder = BottomUpRegionFolder {
            tcx: infcx.tcx,
            fldop_t: |ty| {
                match ty.sty {
                    TyRef(region, tm) if region.needs_infer() => {
                        infcx.tcx.mk_ref(tcx.types.re_erased, tm)
                    },
                    TyInfer(_) => tcx.mk_ty(TyError),
                    _ => ty,
                }
            },
            fldop_r: |reg| {
                if reg.needs_infer() {
                    tcx.types.re_erased
                } else {
                    reg
                }
            },
        };

        if let Err(err) = error {
            let region_maps = RegionMaps::new();
            let mut free_regions = FreeRegionMap::new();
            free_regions.relate_free_regions_from_predicates(new_param_env.caller_bounds);
            infcx.resolve_regions_and_report_errors(new_def_id, &region_maps, &free_regions);

            let err = infcx.resolve_type_vars_if_possible(&err);
            let err = err.fold_with(&mut folder).lift_to_tcx(tcx).unwrap();

            changes.add_change(TypeChanged { error: err }, old_def_id, None);
        }

        let old_param_env = if let Some(env) =
            to_new.translate_param_env(old_def_id, tcx.param_env(old_def_id))
        {
            env
        } else {
            return;
        };
        let mut bound_cx = BoundContext::new(&infcx, old_param_env);
        bound_cx.register(new_def_id, new_substs);

        if let Some(errors) = bound_cx.get_errors() {
            for err in &errors {
                let pred = infcx.resolve_type_vars_if_possible(&err.obligation.predicate);
                let pred = pred.fold_with(&mut folder);

                let err_type = BoundsTightened {
                    pred: pred.lift_to_tcx(tcx).unwrap(),
                };

                changes.add_change(err_type, old_def_id, Some(tcx.def_span(old_def_id)));
            }
        } else {
            let new_param_env_trans = if let Some(env) =
                to_old.translate_param_env(new_def_id, tcx.param_env(new_def_id))
            {
                env
            } else {
                return;
            };
            let mut rev_bound_cx = BoundContext::new(&infcx, new_param_env_trans);
            rev_bound_cx.register(old_def_id, substs);

            if let Some(errors) = rev_bound_cx.get_errors() {
                for err in &errors {
                    let pred = infcx.resolve_type_vars_if_possible(&err.obligation.predicate);
                    let pred = pred.fold_with(&mut folder);

                    let err_type = BoundsLoosened {
                        pred: pred.lift_to_tcx(tcx).unwrap(),
                    };

                    changes.add_change(err_type, old_def_id, Some(tcx.def_span(old_def_id)));
                }
            }
        }
    });
}

/// Compare two implementations and indicate whether the target one is compatible with the
/// original one.
fn match_trait_impl<'a, 'tcx>(id_mapping: &IdMapping,
                              tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              orig_def_id: DefId) -> bool {
    let trans = if id_mapping.in_old_crate(orig_def_id) {
        TranslationContext::target_new(tcx, id_mapping, false)
    } else if id_mapping.in_new_crate(orig_def_id) {
        TranslationContext::target_old(tcx, id_mapping, false)
    } else {
        // not reached, but apparently we don't care.
        return true;
    };

    debug!("matching: {:?}", orig_def_id);

    tcx.infer_ctxt().enter(|infcx| {
        let old_param_env = if let Some(env) =
            trans.translate_param_env(orig_def_id, tcx.param_env(orig_def_id))
        {
            env
        } else {
            return false;
        };

        debug!("env: {:?}", old_param_env);

        let orig = tcx
            .impl_trait_ref(orig_def_id)
            .unwrap();
        debug!("trait ref: {:?}", orig);
        debug!("translated ref: {:?}", trans.translate_trait_ref(orig_def_id, &orig));

        let mut bound_cx = BoundContext::new(&infcx, old_param_env);
        bound_cx.register_trait_ref(trans.translate_trait_ref(orig_def_id, &orig));
        bound_cx.get_errors().is_none()
    })
}

/// Compare an item pair in two inherent implementations and indicate whether the target one is
/// compatible with the original one.
fn match_inherent_impl<'a, 'tcx>(id_mapping: &IdMapping,
                                 tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                 trans: &TranslationContext<'a, 'tcx, 'tcx>,
                                 orig_impl_def_id: DefId,
                                 orig_item_def_id: DefId,
                                 target_impl_def_id: DefId,
                                 target_item_def_id: DefId) -> bool {
    true
}

/// The context in which bounds analysis happens.
pub struct BoundContext<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    /// The inference context to use.
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    /// The fulfillment context to use.
    fulfill_cx: FulfillmentContext<'tcx>,
    /// The param env to be assumed.
    given_param_env: ParamEnv<'tcx>,
}

impl<'a, 'gcx, 'tcx> BoundContext<'a, 'gcx, 'tcx> {
    /// Construct a new bound context.
    pub fn new(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>, given_param_env: ParamEnv<'tcx>) -> Self {
        BoundContext {
            infcx: infcx,
            fulfill_cx: FulfillmentContext::new(),
            given_param_env: given_param_env,
        }
    }

    /// Register the bounds of an item.
    pub fn register(&mut self, checked_def_id: DefId, substs: &Substs<'tcx>) {
        use rustc::traits::{normalize, Normalized, SelectionContext};

        let cause = ObligationCause::dummy();
        let mut selcx = SelectionContext::new(self.infcx);
        let predicates =
            self.infcx
                .tcx
                .predicates_of(checked_def_id)
                .instantiate(self.infcx.tcx, substs);
        let Normalized { value, obligations } =
            normalize(&mut selcx, self.given_param_env, cause.clone(), &predicates);

        for obligation in obligations {
            self.fulfill_cx.register_predicate_obligation(self.infcx, obligation);
        }

        for predicate in value.predicates {
            let obligation = Obligation::new(cause.clone(), self.given_param_env, predicate);
            self.fulfill_cx.register_predicate_obligation(self.infcx, obligation);
        }
    }

    /// Register the trait bound represented by a `TraitRef`.
    pub fn register_trait_ref(&mut self, checked_trait_ref: TraitRef<'tcx>) {
        use rustc::ty::{Binder, Predicate, TraitPredicate};

        let predicate = Predicate::Trait(Binder(TraitPredicate {
            trait_ref: checked_trait_ref,
        }));
        let obligation =
            Obligation::new(ObligationCause::dummy(), self.given_param_env, predicate);
        self.fulfill_cx.register_predicate_obligation(self.infcx, obligation);
    }

    /// Return inference errors, if any.
    pub fn get_errors(&mut self) -> Option<Vec<FulfillmentError<'tcx>>> {
        if let Err(err) = self.fulfill_cx.select_all_or_error(self.infcx) {
            debug!("err: {:?}", err);
            Some(err)
        } else {
            None
        }
    }
}

pub struct TypeComparisonContext<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    id_mapping: &'a IdMapping,
    forward_trans: TranslationContext<'a, 'gcx, 'tcx>,
    backward_trans: TranslationContext<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> TypeComparisonContext<'a, 'gcx, 'tcx> {
    pub fn target_new(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>, id_mapping: &'a IdMapping) -> Self {
        TypeComparisonContext {
            infcx: infcx,
            id_mapping: id_mapping,
            forward_trans: TranslationContext::target_new(infcx.tcx, id_mapping, false),
            backward_trans: TranslationContext::target_old(infcx.tcx, id_mapping, false),
        }
    }

    pub fn target_old(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>, id_mapping: &'a IdMapping) -> Self {
        TypeComparisonContext {
            infcx: infcx,
            id_mapping: id_mapping,
            forward_trans: TranslationContext::target_old(infcx.tcx, id_mapping, false),
            backward_trans: TranslationContext::target_new(infcx.tcx, id_mapping, false),
        }
    }

    fn compute_target_infer_substs(&self, target_def_id: DefId) -> &Substs<'tcx> {
        use syntax_pos::DUMMY_SP;

        let has_self = self.infcx.tcx.generics_of(target_def_id).has_self;

        Substs::for_item(self.infcx.tcx, target_def_id, |def, _| {
            self.infcx.region_var_for_def(DUMMY_SP, def)
        }, |def, substs| {
            if def.index == 0 && has_self { // `Self` is special
                self.infcx.tcx.mk_param_from_def(def)
            } else {
                self.infcx.type_var_for_def(DUMMY_SP, def, substs)
            }
        })
    }

    fn compute_target_default_substs(&self, target_def_id: DefId) -> &Substs<'tcx> {
        use rustc::ty::ReEarlyBound;

        Substs::for_item(self.infcx.tcx, target_def_id, |def, _| {
            self.infcx.tcx.mk_region(ReEarlyBound(def.to_early_bound_region_data()))
        }, |def, _| if self.id_mapping.is_non_mapped_defaulted_type_param(&def.def_id) {
            self.infcx.tcx.type_of(def.def_id)
        } else {
            self.infcx.tcx.mk_param_from_def(def)
        })
    }
}
