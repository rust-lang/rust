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
use rustc::ty::{AssociatedItem, Ty, TyCtxt};
use rustc::ty::Visibility::Public;
use rustc::ty::fold::{BottomUpFolder, TypeFoldable};
use rustc::ty::subst::{Subst, Substs};

use semcheck::changes::BinaryChangeType;
use semcheck::changes::BinaryChangeType::*;
use semcheck::changes::ChangeSet;
use semcheck::mapping::{IdMapping, NameMapping};
use semcheck::mismatch::Mismatch;

use std::collections::{BTreeMap, HashSet, VecDeque};

/// The main entry point to our analysis passes.
///
/// Set up the necessary data structures and run the analysis passes.
pub fn run_analysis<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, old: DefId, new: DefId)
    -> ChangeSet<'tcx>
{
    let mut changes = Default::default();
    let mut id_mapping = Default::default();

    // first pass
    diff_structure(&mut changes, &mut id_mapping, tcx, old, new);

    // second pass
    {
        let mut mismatch = Mismatch::new(tcx, old.krate, &mut id_mapping);
        mismatch.process();
    }

    // third pass
    for &(old, new) in id_mapping.toplevel_values() {
        diff_bounds(&mut changes, tcx, old.def_id(), new.def_id());
    }

    // fourth pass
    for &(old, new) in id_mapping.toplevel_values() {
        diff_types(&mut changes, &id_mapping, tcx, old, new);
    }

    changes
}

// Below functions constitute the first pass of analysis, in which module structure, ADT
// structure, public and private status of items, and generics are examined for changes.

/// Given two crate root modules, compare their exports and their structure.
///
/// Traverse the two root modules in an interleaved manner, matching up pairs of modules
/// from the two crate versions and compare for changes. Matching children get processed
/// in the same fashion.
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

    mod_queue.push_back((old, new, Public, Public));

    while let Some((old_did, new_did, old_vis, new_vis)) = mod_queue.pop_front() {
        children.add(cstore.item_children(old_did, tcx.sess),
                     cstore.item_children(new_did, tcx.sess));

        for items in children.drain() {
            match items {
                (Some(o), Some(n)) => {
                    if let (Mod(o_did), Mod(n_did)) = (o.def, n.def) {
                        if visited.insert((o_did, n_did)) {
                            let o_vis = if old_vis == Public {
                                cstore.visibility(o_did)
                            } else {
                                old_vis
                            };
                            let n_vis = if new_vis == Public {
                                cstore.visibility(n_did)
                            } else {
                                new_vis
                            };

                            if o_vis != n_vis {
                                changes.new_binary(o.def.def_id(), o.ident.name, n.span, true);

                                if o_vis == Public && n_vis != Public {
                                    changes.add_binary(ItemMadePrivate, o_did, None);
                                } else if o_vis != Public && n_vis == Public {
                                    changes.add_binary(ItemMadePublic, o_did, None);
                                }
                            }

                            mod_queue.push_back((o_did, n_did, o_vis, n_vis));
                        }
                    } else if id_mapping.add_export(o.def, n.def) {
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
                        changes.new_binary(o.def.def_id(), o.ident.name, n.span, output);

                        if o_vis == Public && n_vis != Public {
                            changes.add_binary(ItemMadePrivate, o_def_id, None);
                        } else if o_vis != Public && n_vis == Public {
                            changes.add_binary(ItemMadePublic, o_def_id, None);
                        }

                        // TODO: extend as needed.
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
                            // ADTs for now
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
                                diff_traits(changes, id_mapping, tcx, o_def_id, n_def_id);
                            },
                            // non-matching item pair - register the difference and abort
                            _ => {
                                changes.add_binary(KindDifference, o_def_id, None);
                            },
                        }
                    }
                }
                (Some(o), None) => {
                    if old_vis == Public && cstore.visibility(o.def.def_id()) == Public {
                        changes.new_removal(o);
                    }
                }
                (None, Some(n)) => {
                    if new_vis == Public && cstore.visibility(n.def.def_id()) == Public {
                        changes.new_addition(n);
                    }
                }
                (None, None) => unreachable!(),
            }
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
        changes.add_binary(FnConstChanged { now_const: new_const }, old_def_id, None);
    }
}

/// Given two method items, perform structural checks.
fn diff_method(changes: &mut ChangeSet, tcx: TyCtxt, old: AssociatedItem, new: AssociatedItem) {
    if old.method_has_self_argument != new.method_has_self_argument {
        changes.add_binary(MethodSelfChanged { now_self: new.method_has_self_argument },
                           old.def_id,
                           None);
    }

    diff_fn(changes, tcx, Def::Method(old.def_id), Def::Method(new.def_id));
}

/// Given two ADT items, perform structural checks.
///
/// This establishes the needed correspondence relationship between non-toplevel items such as
/// enum variants, struct fields etc.
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
                id_mapping.add_subitem(old_def_id, old.did, new.did);

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
                    changes.add_binary(c, old_def_id, Some(tcx.def_span(new.did)));

                    continue;
                }

                for items2 in fields.values() {
                    match *items2 {
                        (Some(o), Some(n)) => {
                            id_mapping.add_subitem(old_def_id, o.did, n.did);

                            if o.vis != Public && n.vis == Public {
                                changes.add_binary(ItemMadePublic,
                                                   old_def_id,
                                                   Some(tcx.def_span(n.did)));
                            } else if o.vis == Public && n.vis != Public {
                                changes.add_binary(ItemMadePrivate,
                                                   old_def_id,
                                                   Some(tcx.def_span(n.did)));
                            }
                        },
                        (Some(o), None) => {
                            let c = VariantFieldRemoved {
                                public: o.vis == Public,
                                total_public: total_public
                            };
                            changes.add_binary(c, old_def_id, Some(tcx.def_span(o.did)));
                        },
                        (None, Some(n)) => {
                            let c = VariantFieldAdded {
                                public: n.vis == Public,
                                total_public: total_public
                            };
                            changes.add_binary(c, old_def_id, Some(tcx.def_span(n.did)));
                        },
                        (None, None) => unreachable!(),
                    }
                }

                fields.clear();
            },
            (Some(old), None) => {
                changes.add_binary(VariantRemoved, old_def_id, Some(tcx.def_span(old.did)));
            },
            (None, Some(new)) => {
                changes.add_binary(VariantAdded, old_def_id, Some(tcx.def_span(new.did)));
            },
            (None, None) => unreachable!(),
        }
    }
}

/// Given two trait items, perform structural checks.
///
/// This establishes the needed correspondence relationship between non-toplevel items found in
/// the trait definition.
fn diff_traits(changes: &mut ChangeSet,
               id_mapping: &mut IdMapping,
               tcx: TyCtxt,
               old: DefId,
               new: DefId) {
    let _old_unsafety = tcx.trait_def(old).unsafety;
    let _new_unsafety = tcx.trait_def(new).unsafety;

    let mut items = BTreeMap::new();

    for old_did in tcx.associated_item_def_ids(old).iter() {
        let item = tcx.associated_item(*old_did);
        items.entry(item.name).or_insert((None, None)).0 =
            tcx.describe_def(*old_did).map(|d| (d, item));
    }

    for new_did in tcx.associated_item_def_ids(new).iter() {
        let item = tcx.associated_item(*new_did);
        items.entry(item.name).or_insert((None, None)).1 =
            tcx.describe_def(*new_did).map(|d| (d, item));

    }

    for (name, item_pair) in &items {
        match *item_pair {
            (Some((old_def, old_item)), Some((new_def, new_item))) => {
                id_mapping.add_trait_item(old_def, new_def);
                changes.new_binary(old_def.def_id(),
                                   *name,
                                   tcx.def_span(new_def.def_id()),
                                   true);
                diff_method(changes, tcx, old_item, new_item);
            },
            (Some((_, old_item)), None) => {
                let change_type = TraitItemRemoved {
                    defaulted: old_item.defaultness.has_value(),
                };
                changes.add_binary(change_type , old, Some(tcx.def_span(old_item.def_id)));
            },
            (None, Some((_, new_item))) => {
                let change_type = TraitItemAdded {
                    defaulted: new_item.defaultness.has_value(),
                };
                changes.add_binary(change_type , old, Some(tcx.def_span(new_item.def_id)));
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
            (Some(_ /* old_region */), None) => {
                found.push(RegionParameterRemoved);
            },
            (None, Some(_ /* new_region */)) => {
                found.push(RegionParameterAdded);
            },
            (Some(_), Some(_)) => (),
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
            },
            (Some(old_type), None) => {
                found.push(TypeParameterRemoved { defaulted: old_type.has_default });
            },
            (None, Some(new_type)) => { // FIXME: is_fn could be used in a more elegant fashion
                found.push(TypeParameterAdded { defaulted: new_type.has_default || is_fn });
                if new_type.has_default {
                    id_mapping.add_defaulted_type_param(new_type.def_id);
                }
            },
            (None, None) => unreachable!(),
        }
    }

    for change_type in found.drain(..) {
        changes.add_binary(change_type, old, None);
    }
}

// Below functions constitute the third pass of analysis, in which parameter bounds of matching
// items are compared for changes and used to determine matching relationships between items not
// being exported.

/// Given two items, compare the bounds on their type and region parameters.
fn diff_bounds<'a, 'tcx>(_changes: &mut ChangeSet,
                         _tcx: TyCtxt<'a, 'tcx, 'tcx>,
                         _old: DefId,
                         _new: DefId)
    -> (Vec<BinaryChangeType<'tcx>>, Vec<(DefId, DefId)>)
{
    /* let res = Default::default();

    let old_preds = tcx.predicates_of(old).predicates;
    let new_preds = tcx.predicates_of(new).predicates;

    res */

    Default::default()
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
    use rustc::ty::Binder;

    let old_def_id = old.def_id();
    let new_def_id = new.def_id();

    if changes.item_breaking(old_def_id) {
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
            let old_fn_sig =
                Binder(fold_to_new(id_mapping,
                                   tcx,
                                   tcx.type_of(old_def_id).fn_sig(tcx).skip_binder()));
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
            if let Some(children) = id_mapping.children_values(old_def_id) {
                for (o_did, n_did) in children {
                    let o_ty = tcx.type_of(o_did);
                    let n_ty = tcx.type_of(n_did);

                    cmp_types(changes, id_mapping, tcx, old_def_id, new_def_id, o_ty, n_ty);
                }
            }
        },
        _ => (),
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
    use syntax_pos::DUMMY_SP;
    use rustc::ty::{Lift, ReEarlyBound};

    let substs = Substs::identity_for_item(tcx, new_def_id);

    let old = fold_to_new(id_mapping, tcx, &old.subst(tcx, substs));

    tcx.infer_ctxt().enter(|infcx| {
        let new_substs = if new.is_fn() {
            infcx.fresh_substs_for_item(DUMMY_SP, new_def_id)
        } else {
            Substs::for_item(tcx, new_def_id, |def, _| {
                tcx.mk_region(ReEarlyBound(def.to_early_bound_region_data()))
            }, |def, _| if id_mapping.is_defaulted_type_param(&def.def_id) {
                tcx.type_of(def.def_id)
            } else {
                tcx.mk_param_from_def(def)
            })
        };

        let new = new.subst(infcx.tcx, new_substs);

        if let Err(err) = infcx.can_eq(tcx.param_env(new_def_id), old, new) {
            let err = match infcx.fully_resolve(&err) {
                Ok(res) => res,
                Err(err) => panic!("err: {:?}", err),
            };
            changes.add_binary(TypeChanged { error: err.lift_to_tcx(tcx).unwrap() },
                               old_def_id,
                               None);
        }
    });
}

/// Fold a type of an old item to be comparable with a new type.
fn fold_to_new<'a, 'tcx, T>(id_mapping: &IdMapping, tcx: TyCtxt<'a, 'tcx, 'tcx>, old: &T) -> T
    where T: TypeFoldable<'tcx>
{
    use rustc::ty::{AdtDef, Binder, ExistentialProjection, ExistentialTraitRef};
    use rustc::ty::ExistentialPredicate::*;
    use rustc::ty::TypeVariants::*;

    old.fold_with(&mut BottomUpFolder { tcx: tcx, fldop: |ty| {
        match ty.sty {
            TyAdt(&AdtDef { ref did, .. }, substs) if id_mapping.contains_id(*did) => {
                let new_did = id_mapping.get_new_id(*did);
                let new_adt = tcx.adt_def(new_did);
                tcx.mk_adt(new_adt, substs)
            },
            TyDynamic(preds, region) => {
                let new_preds = tcx.mk_existential_predicates(preds.iter().map(|p| {
                    match *p.skip_binder() {
                        Trait(ExistentialTraitRef { def_id: did, substs }) => {
                            let new_did = if id_mapping.contains_id(did) {
                                id_mapping.get_new_id(did)
                            } else {
                                did
                            };

                            Trait(ExistentialTraitRef {
                                def_id: new_did,
                                substs: substs
                            })
                        },
                        Projection(ExistentialProjection { trait_ref, item_name, ty }) => {
                            let ExistentialTraitRef { def_id: did, substs } = trait_ref;
                            let new_did = if id_mapping.contains_id(did) {
                                id_mapping.get_new_id(did)
                            } else {
                                did
                            };

                            Projection(ExistentialProjection {
                                trait_ref: ExistentialTraitRef {
                                    def_id: new_did,
                                    substs: substs,
                                },
                                item_name: item_name,
                                ty: ty,
                            })

                        },
                        AutoTrait(did) => {
                            if id_mapping.contains_id(did) {
                                AutoTrait(id_mapping.get_new_id(did))
                            } else {
                                AutoTrait(did)
                            }
                        },
                    }
                }));

                tcx.mk_dynamic(Binder(new_preds), region)
            },
            _ => ty,
        }
    }})
}
