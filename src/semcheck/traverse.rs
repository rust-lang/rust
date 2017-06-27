use semcheck::changes::BinaryChangeType;
use semcheck::changes::BinaryChangeType::*;
use semcheck::changes::ChangeSet;
use semcheck::id_mapping::{get_namespace, IdMapping};
use semcheck::mismatch::Mismatch;

use rustc::hir::def::CtorKind;
use rustc::hir::def::Export;
use rustc::hir::def_id::DefId;
use rustc::ty::TyCtxt;
use rustc::ty::Visibility::Public;
use rustc::ty::fold::{BottomUpFolder, TypeFoldable};
use rustc::ty::relate::TypeRelation;
use rustc::ty::subst::{Subst, Substs};

use std::collections::{HashMap, HashSet, VecDeque};

/// The main entry point to our analysis passes.
///
/// Set up the necessary data structures and run the analysis passes.
pub fn run_analysis(tcx: TyCtxt, old: DefId, new: DefId) -> ChangeSet {
    let mut changes = Default::default();
    let mut id_mapping = Default::default();

    // first pass
    diff_structure(&mut changes, &mut id_mapping, tcx, old, new);

    // second pass
    {
        let item_queue: VecDeque<_> =
            id_mapping
                .toplevel_mapping
                .values()
                .map(|&(_, old, new)| (old.def.def_id(), new.def.def_id()))
                .collect();

        let mut mismatch = Mismatch::new(tcx, item_queue);

        while let Some((old_did, new_did)) = mismatch.item_queue.pop_front() {
            let _ = mismatch.tys(tcx.type_of(old_did), tcx.type_of(new_did));

            if !id_mapping.contains_id(old_did) {
                // println!("adding mapping: {:?} => {:?}", old_did, new_did);
                id_mapping.add_item(old_did, new_did);
            }
        }
    }

    // third pass
    for &(_, old, new) in id_mapping.toplevel_mapping.values() {
        diff_bounds(&mut changes, tcx, old.def.def_id(), new.def.def_id());
    }

    // fourth pass
    for &(_, old, new) in id_mapping.toplevel_mapping.values() {
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
fn diff_structure(changes: &mut ChangeSet,
                  id_mapping: &mut IdMapping,
                  tcx: TyCtxt,
                  old: DefId,
                  new: DefId) {
    use rustc::hir::def::Def::*;

    let cstore = &tcx.sess.cstore;
    let mut visited = HashSet::new();
    let mut children = HashMap::new();
    let mut mod_queue = VecDeque::new();

    mod_queue.push_back((old, new, Public, Public));

    while let Some((old_did, new_did, old_vis, new_vis)) = mod_queue.pop_front() {
        let mut c_old = cstore.item_children(old_did, tcx.sess);
        let mut c_new = cstore.item_children(new_did, tcx.sess);

        // TODO: refactor this to avoid storing tons of `Namespace` values.
        for child in c_old.drain(..) {
            let key = (get_namespace(&child.def), child.ident.name);
            children.entry(key).or_insert((None, None)).0 = Some(child);
        }

        for child in c_new.drain(..) {
            let key = (get_namespace(&child.def), child.ident.name);
            children.entry(key).or_insert((None, None)).1 = Some(child);
        }

        for (_, items) in children.drain() {
            match items {
                (Some(Export { def: Mod(o), .. }), Some(Export { def: Mod(n), .. })) => {
                    if visited.insert((o, n)) {
                        let o_vis = if old_vis == Public {
                            cstore.visibility(o)
                        } else {
                            old_vis
                        };
                        let n_vis = if new_vis == Public {
                            cstore.visibility(n)
                        } else {
                            new_vis
                        };

                        if o_vis != n_vis {
                            // TODO: ugly
                            changes.new_binary(items.0.unwrap(), items.1.unwrap(), true);

                            if o_vis == Public && n_vis != Public {
                                changes.add_binary(ItemMadePrivate, o, None);
                            } else if o_vis != Public && n_vis == Public {
                                changes.add_binary(ItemMadePublic, o, None);
                            }
                        }

                        mod_queue.push_back((o, n, o_vis, n_vis));
                    }
                }
                (Some(o), Some(n)) => {
                    if !id_mapping.add_export(o, n) {
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
                        changes.new_binary(o, n, output);

                        if o_vis == Public && n_vis != Public {
                            changes.add_binary(ItemMadePrivate, o_def_id, None);
                        } else if o_vis != Public && n_vis == Public {
                            changes.add_binary(ItemMadePublic, o_def_id, None);
                        }

                        // TODO: extend as needed.
                        match (o.def, n.def) {
                            // (matching) things we don't care about (for now)
                            (Mod(_), Mod(_)) |
                            (Trait(_), Trait(_)) |
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
                                diff_fn(changes, tcx, o, n);
                            },
                            (TyAlias(_), TyAlias(_)) => {
                                let mut generics_changes =
                                    diff_generics(tcx, o_def_id, n_def_id);
                                for change_type in generics_changes.drain(..) {
                                    changes.add_binary(change_type, o_def_id, None);
                                }
                            },
                            // ADTs for now
                            (Struct(_), Struct(_)) |
                            (Union(_), Union(_)) |
                            (Enum(_), Enum(_)) => {
                                let mut generics_changes =
                                    diff_generics(tcx, o_def_id, n_def_id);
                                for change_type in generics_changes.drain(..) {
                                    changes.add_binary(change_type, o_def_id, None);
                                }

                                diff_adts(changes, id_mapping, tcx, o, n);
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
fn diff_fn(changes: &mut ChangeSet,
           tcx: TyCtxt,
           old: Export,
           new: Export) {
    use rustc::hir::Unsafety::Unsafe;
    use rustc::ty::TypeVariants::*;

    let old_def_id = old.def.def_id();
    let new_def_id = new.def.def_id();

    let old_ty = tcx.type_of(old_def_id);
    let new_ty = tcx.type_of(new_def_id);

    let (old_sig, new_sig) = match (&old_ty.sty, &new_ty.sty) {
        (&TyFnDef(_, _, ref o), &TyFnDef(_, _, ref n)) |
        (&TyFnPtr(ref o), &TyFnPtr(ref n)) => (o.skip_binder(), n.skip_binder()),
        _ => return,
    };

    if old_sig.variadic != new_sig.variadic {
        changes.add_binary(FnVariadicChanged, old_def_id, None);
    }

    if old_sig.unsafety != new_sig.unsafety {
        let change = FnUnsafetyChanged { now_unsafe: new_sig.unsafety == Unsafe };
        changes.add_binary(change, old_def_id, None);
    }

    if old_sig.abi != new_sig.abi {
        // TODO: more sophisticatd comparison
        changes.add_binary(FnAbiChanged, old_def_id, None);
    }

    if old_sig.inputs_and_output.len() != new_sig.inputs_and_output.len() {
        changes.add_binary(FnArityChanged, old_def_id, None);
    }
}

/// Given two ADT items, perform structural checks.
///
/// This establishes the needed correspondence relationship between non-toplevel items such as
/// enum variants, struct fields etc.
fn diff_adts(changes: &mut ChangeSet,
             id_mapping: &mut IdMapping,
             tcx: TyCtxt,
             old: Export,
             new: Export) {
    use rustc::hir::def::Def::*;

    let old_def_id = old.def.def_id();
    let new_def_id = new.def.def_id();

    let (old_def, new_def) = match (old.def, new.def) {
        (Struct(_), Struct(_)) |
        (Union(_), Union(_)) |
        (Enum(_), Enum(_)) => (tcx.adt_def(old_def_id), tcx.adt_def(new_def_id)),
        _ => return,
    };

    let mut variants = HashMap::new();
    let mut fields = HashMap::new();

    for variant in &old_def.variants {
        variants.entry(variant.name).or_insert((None, None)).0 = Some(variant);
    }

    for variant in &new_def.variants {
        variants.entry(variant.name).or_insert((None, None)).1 = Some(variant);
    }

    for (_, items) in variants.drain() {
        match items {
            (Some(old), Some(new)) => {
                id_mapping.add_item(old.did, new.did);

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
                    changes.add_binary(c, old_def_id, Some(tcx.def_span(old.did)));
                }

                for (_, items2) in fields.drain() {
                    match items2 {
                        (Some(o), Some(n)) => {
                            id_mapping.add_item(o.did, n.did);

                            if o.vis != Public && n.vis == Public {
                                changes.add_binary(ItemMadePublic,
                                                   old_def_id,
                                                   Some(tcx.def_span(o.did)));
                            } else if o.vis == Public && n.vis != Public {
                                changes.add_binary(ItemMadePrivate,
                                                   old_def_id,
                                                   Some(tcx.def_span(o.did)));
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

/// Given two items, compare their type and region parameter sets.
fn diff_generics(tcx: TyCtxt, old: DefId, new: DefId) -> Vec<BinaryChangeType> {
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

// Below functions constitute the third pass of analysis, in which parameter bounds of matching
// items are compared for changes and used to determine matching relationships between items not
// being exported.

/// Given two items, compare the bounds on their type and region parameters.
fn diff_bounds(_changes: &mut ChangeSet, _tcx: TyCtxt, _old: DefId, _new: DefId)
    -> (Vec<BinaryChangeType>, Vec<(DefId, DefId)>)
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
fn diff_types(changes: &mut ChangeSet,
              id_mapping: &IdMapping,
              tcx: TyCtxt,
              old: Export,
              new: Export) {
    use rustc::ty::AdtDef;
    use rustc::ty::TypeVariants::*;

    if changes.item_breaking(old.def.def_id()) {
        return;
    }

    let old_def_id = old.def.def_id();
    let new_def_id = new.def.def_id();

    let new_ty = tcx.type_of(new_def_id);
    let old_ty = tcx.type_of(old_def_id).subst(tcx, Substs::identity_for_item(tcx, new_def_id));

    let old_ty_cmp = old_ty.fold_with(&mut BottomUpFolder { tcx: tcx, fldop: |ty| {
        match ty.sty {
            TyAdt(&AdtDef { ref did, .. }, substs) => {
                let new_did = id_mapping.get_new_id(*did);
                let new_adt = tcx.adt_def(new_did);
                tcx.mk_adt(new_adt, substs)
            },
            /* TyDynamic(predicates, region) => {

            }, TyClosure, TyRef (because of region?), TyProjection, TyAnon
            TyProjection(projection_ty) => {

            }, */
            _ => ty,
        }
    }});

    // println!("old_ty: {:?}", old_ty_cmp);
    // println!("new_ty: {:?}", new_ty);

    if let Err(err) = tcx.global_tcx().infer_ctxt()
        .enter(|infcx| infcx.can_eq(tcx.param_env(new_def_id), old_ty_cmp, new_ty))
    {
        // println!("diff: {}", err);
        changes.add_binary(FieldTypeChanged(format!("{}", err)), old_def_id, None);
    }
}
