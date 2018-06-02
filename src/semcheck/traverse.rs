//! The traversal logic collecting changes in between crate versions.
//!
//! The changes get collected in multiple passes, and recorded in a `ChangeSet`.
//! The initial pass matches items by name in the module hierarchy, registering item removal
//! and addition, as well as structural changes to ADTs, type- or region parameters, and
//! function signatures. The second pass then proceeds to find non-public items that are named
//! differently, yet are compatible in their usage. The third pass now uses the information
//! collected in the previous passes to compare the types and/or trait bounds of all item pairs
//! that have been matched. Trait and inherent impls can't be matched by name, and are processed
//! in a fourth pass that uses trait bounds to find matching impls.

use rustc::hir::def::{CtorKind, Def, Export};
use rustc::hir::def_id::DefId;
use rustc::ty::{AssociatedItem, Generics, GenericParamDef, GenericParamDefKind, Ty, TyCtxt};
use rustc::ty::subst::{Subst, Substs};
use rustc::ty::Visibility;
use rustc::ty::Visibility::Public;

use semcheck::changes::ChangeType;
use semcheck::changes::ChangeType::*;
use semcheck::changes::ChangeSet;
use semcheck::mapping::{IdMapping, NameMapping};
use semcheck::mismatch::MismatchRelation;
use semcheck::translate::TranslationContext;
use semcheck::typeck::{BoundContext, TypeComparisonContext};

use std::collections::{BTreeMap, HashSet, VecDeque};

/// The main entry point to our analysis passes.
///
/// Set up the necessary data structures and run the analysis passes and call the actual passes.
pub fn run_analysis<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, old: DefId, new: DefId)
    -> ChangeSet<'tcx>
{
    let mut changes = Default::default();
    let mut id_mapping = IdMapping::new(old.krate, new.krate);

    // first pass
    debug!("first pass started");
    diff_structure(&mut changes, &mut id_mapping, tcx, old, new);

    // second pass
    debug!("second pass started");
    {
        let mut mismatch = MismatchRelation::new(tcx, &mut id_mapping);
        debug!("constructed mismatch relation");
        mismatch.process();
    }

    // third pass
    debug!("third pass started");
    for (old, new) in id_mapping.items() {
        diff_types(&mut changes, &id_mapping, tcx, old, new);
    }

    // fourth pass on impls
    debug!("fourth pass started");
    diff_inherent_impls(&mut changes, &id_mapping, tcx);
    diff_trait_impls(&mut changes, &id_mapping, tcx);

    changes
}

// Below functions constitute the first pass of analysis, in which module structure, ADT
// structure, public and private status of items, and generics are examined for changes.

/// Given two crate root modules, compare their exports and structure.
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

    // Get the visibility of the inner item, given the outer item's visibility.
    fn get_vis(outer_vis: Visibility, def: Export) -> Visibility {
        if outer_vis == Public {
            def.vis
        } else {
            outer_vis
        }
    }

    let mut visited = HashSet::new();
    let mut children = NameMapping::default();
    let mut mod_queue = VecDeque::new();
    let mut traits = Vec::new();
    // Additions and removals are processed with a delay to avoid creating multiple path change
    // entries. This is necessary, since the order in which added or removed paths are found wrt
    // each other and their item's definition can't be relied upon.
    let mut removals = Vec::new();
    let mut additions = Vec::new();

    // Start off with the root module pair.
    mod_queue.push_back((old, new, Public, Public));

    // Pull a matched module pair from the queue, with the modules' global visibility.
    while let Some((old_def_id, new_def_id, old_vis, new_vis)) = mod_queue.pop_front() {
        children.add(tcx.item_children(old_def_id).to_vec(), // TODO: clean up
                     tcx.item_children(new_def_id).to_vec());

        for items in children.drain() {
            match items {
                // an item pair is found
                (Some(o), Some(n)) => {
                    if let (Mod(o_def_id), Mod(n_def_id)) = (o.def, n.def) {
                        if visited.insert((o_def_id, n_def_id)) {
                            let o_vis = get_vis(old_vis, o);
                            let n_vis = get_vis(new_vis, n);

                            if o_vis != n_vis {
                                changes.new_change(o_def_id,
                                                   n_def_id,
                                                   o.ident.name,
                                                   tcx.def_span(o_def_id),
                                                   tcx.def_span(n_def_id),
                                                   true);

                                // this seemingly overly complex condition is needed to handle
                                // `Restricted` visibility correctly.
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
                        let o_vis = get_vis(old_vis, o);
                        let n_vis = get_vis(new_vis, n);

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
                            // matching items we don't care about because they are either
                            // impossible to encounter at this stage (Mod, AssociatedTy, PrimTy,
                            // TyParam, SelfTy, StructCtor, VariantCtor, AssociatedConst, Local,
                            // Upvar, Label, Variant, Method, Err), whose analysis is out scope
                            // for us (GlobalAsm, Macro), or which don't requite further analysis
                            // at this stage (Const)
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
                            (Method(_), Method(_)) |
                            (Err, Err) => {},
                            // statics are subject to mutability comparison
                            (Static(_, old_mut), Static(_, new_mut)) => {
                                if old_mut != new_mut {
                                    let change_type = StaticMutabilityChanged {
                                        now_mut: new_mut,
                                    };

                                    changes.add_change(change_type, o_def_id, None);
                                }
                            },
                            // functions can declare generics and have structural properties
                            // that need to be compared
                            (Fn(_), Fn(_)) => {
                                diff_generics(changes,
                                              id_mapping,
                                              tcx,
                                              true,
                                              o_def_id,
                                              n_def_id);
                                diff_fn(changes, tcx, o.def, n.def);
                            },
                            // type aliases can declare generics, too
                            (TyAlias(_), TyAlias(_)) => {
                                diff_generics(changes,
                                              id_mapping,
                                              tcx,
                                              false,
                                              o_def_id,
                                              n_def_id);
                            },
                            // ADTs can declare generics and have lots of structural properties
                            // to check, most notably the number and name of variants and/or
                            // fields
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
                            // trait definitions can declare generics and require us to check
                            // for trait item addition and removal, as well as changes to their
                            // kinds and defaultness
                            (Trait(_), Trait(_)) => {
                                if o_vis != Public {
                                    debug!("private trait: {:?}", o_def_id);
                                    id_mapping.add_private_trait(o_def_id);
                                }

                                if n_vis != Public {
                                    debug!("private trait: {:?}", n_def_id);
                                    id_mapping.add_private_trait(n_def_id);
                                }

                                diff_generics(changes,
                                              id_mapping,
                                              tcx,
                                              false,
                                              o_def_id,
                                              n_def_id);
                                traits.push((o_def_id, n_def_id, output));
                                /* diff_traits(changes,
                                            id_mapping,
                                            tcx,
                                            o_def_id,
                                            n_def_id,
                                            output); */
                            },
                            // a non-matching item pair - register the change and abort further
                            // analysis of it
                            _ => {
                                changes.add_change(KindDifference, o_def_id, None);
                            },
                        }
                    }
                }
                // only an old item is found
                (Some(o), None) => {
                    // struct constructors are weird/hard - let's go shopping!
                    if let StructCtor(_, _) = o.def {
                        continue;
                    }

                    if get_vis(old_vis, o) == Public {
                        // delay the handling of removals until the id mapping is complete
                        removals.push(o);
                    }
                }
                // only a new item is found
                (None, Some(n)) => {
                    // struct constructors are weird/hard - let's go shopping!
                    if let StructCtor(_, _) = n.def {
                        continue;
                    }

                    if get_vis(new_vis, n) == Public {
                        debug!("addition: {:?} ({:?})", new_vis, n);
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

    for (o_def_id, n_def_id, output) in traits {
        diff_traits(changes, id_mapping, tcx, o_def_id, n_def_id, output);
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

    let old_pub = old.vis == Public;
    let new_pub = new.vis == Public;

    if old_pub && !new_pub {
        changes.add_change(ItemMadePrivate, old.def_id, None);
    } else if !old_pub && new_pub {
        changes.add_change(ItemMadePublic, old.def_id, None);
    }

    diff_fn(changes, tcx, Def::Method(old.def_id), Def::Method(new.def_id));
}

/// Given two ADT items, perform structural checks.
///
/// This establishes the needed correspondence between non-toplevel items such as enum variants,
/// struct- and enum fields etc.
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
                    fields.entry(field.ident.name).or_insert((None, None)).0 = Some(field);
                }

                for field in &new.fields {
                    fields.entry(field.ident.name).or_insert((None, None)).1 = Some(field);
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
                            if o.vis == Public && n.vis == Public {
                                id_mapping.add_subitem(old_def_id, o.did, n.did);
                            } else if o.vis != Public && n.vis == Public {
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
    use rustc::ty::subst::UnpackedKind::Type;
    use rustc::ty::{ParamTy, Predicate, TyS, TypeVariants::*};

    debug!("diff_traits: old: {:?}, new: {:?}, output: {:?}", old, new, output);

    let old_unsafety = tcx.trait_def(old).unsafety;
    let new_unsafety = tcx.trait_def(new).unsafety;

    if old_unsafety != new_unsafety {
        let change_type = TraitUnsafetyChanged {
            now_unsafe: new_unsafety == Unsafe,
        };

        changes.add_change(change_type, old, None);
    }

    let mut old_sealed = false;
    let old_param_env = tcx.param_env(old);

    for bound in old_param_env.caller_bounds {
        match *bound {
            Predicate::Trait(pred) => {
                let trait_ref = pred.skip_binder().trait_ref;

                debug!("trait_ref substs (old): {:?}", trait_ref.substs);

                if id_mapping.is_private_trait(&trait_ref.def_id) &&
                        trait_ref.substs.len() == 1 {
                    if let Type(&TyS { sty: TyParam(ParamTy { idx: 0, ..}), ..}) =
                            trait_ref.substs[0].unpack() {
                        old_sealed = true;
                    }
                }
            },
            _ => (),
        }
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
                    sealed_trait: old_sealed,
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
    use rustc::ty::Variance;
    use rustc::ty::Variance::*;
    use std::cmp::max;

    debug!("diff_generics: old: {:?}, new: {:?}", old, new);

    fn diff_variance<'tcx>(old_var: Variance, new_var: Variance) -> Option<ChangeType<'tcx>> {
        match (old_var, new_var) {
            (Covariant, Covariant) |
            (Invariant, Invariant) |
            (Contravariant, Contravariant) |
            (Bivariant, Bivariant) => None,
            (Invariant, _) | (_, Bivariant) => Some(VarianceLoosened),
            (_, Invariant) | (Bivariant, _) => Some(VarianceTightened),
            (Covariant, Contravariant) => Some(VarianceChanged { now_contravariant: true }),
            (Contravariant, Covariant) => Some(VarianceChanged { now_contravariant: false }),
        }
    }

    let mut found = Vec::new();

    let old_gen = tcx.generics_of(old);
    let new_gen = tcx.generics_of(new);

    let old_var = tcx.variances_of(old);
    let new_var = tcx.variances_of(new);

    let old_count = old_gen.own_counts();
    let new_count = new_gen.own_counts();

    // guarantee that the return value's kind is `GenericParamDefKind::Lifetime`
    fn get_region_from_params(gen: &Generics, idx: usize) -> Option<&GenericParamDef> {
        let param = gen.params.get(idx)?;

        match param.kind {
            GenericParamDefKind::Lifetime => Some(param),
            _ => None,
        }
    }

    for i in 0..max(old_count.lifetimes, new_count.lifetimes) {
        match (get_region_from_params(old_gen, i), get_region_from_params(new_gen, i)) {
            (Some(old_region), Some(new_region)) => {
                // type aliases don't have inferred variance, so we have to ignore that.
                if let (Some(old_var), Some(new_var)) = (old_var.get(i), new_var.get(i)) {
                    diff_variance(*old_var, *new_var)
                        .map(|t| found.push(t));
                }

                id_mapping.add_internal_item(old_region.def_id, new_region.def_id);
            },
            (Some(_), None) => {
                found.push(RegionParameterRemoved);
            },
            (None, Some(_)) => {
                found.push(RegionParameterAdded);
            },
            (None, None) => unreachable!(),
        }
    }

    // guarantee that the return value's kind is `GenericParamDefKind::Type`
    fn get_type_from_params(gen: &Generics, idx: usize) -> Option<&GenericParamDef> {
        let param = &gen.params.get(idx)?;

        match param.kind {
            GenericParamDefKind::Type { .. } => Some(param),
            _ => None,
        }
    }

    for i in 0..max(old_count.types, new_count.types) {
        match (get_type_from_params(old_gen, old_count.lifetimes + i),
               get_type_from_params(new_gen, new_count.lifetimes + i)) {
            (Some(old_type), Some(new_type)) => {
                // type aliases don't have inferred variance, so we have to ignore that.
                if let (Some(old_var), Some(new_var)) =
                    (old_var.get(old_count.lifetimes + i), new_var.get(new_count.lifetimes + i))
                {
                    diff_variance(*old_var, *new_var)
                        .map(|t| found.push(t));
                }

                let old_default = match old_type.kind {
                    GenericParamDefKind::Type { has_default, .. } => has_default,
                    _ => unreachable!(),
                };
                let new_default = match new_type.kind {
                    GenericParamDefKind::Type { has_default, .. } => has_default,
                    _ => unreachable!(),
                };

                if old_default && !new_default {
                    found.push(TypeParameterRemoved { defaulted: true });
                    found.push(TypeParameterAdded { defaulted: false });
                } else if !old_default && new_default {
                    found.push(TypeParameterRemoved { defaulted: false });
                    found.push(TypeParameterAdded { defaulted: true });
                }

                debug!("in item {:?} / {:?}:\n  type param pair: {:?}, {:?}",
                       old, new, old_type, new_type);

                id_mapping.add_internal_item(old_type.def_id, new_type.def_id);
                id_mapping.add_type_param(old_type);
                id_mapping.add_type_param(new_type);
            },
            (Some(old_type), None) => {
                let old_default = match old_type.kind {
                    GenericParamDefKind::Type { has_default, .. } => has_default,
                    _ => unreachable!(),
                };

                found.push(TypeParameterRemoved { defaulted: old_default });
                id_mapping.add_type_param(old_type);
                id_mapping.add_non_mapped(old_type.def_id);
            },
            (None, Some(new_type)) => {
                let new_default = match new_type.kind {
                    GenericParamDefKind::Type { has_default, .. } => has_default,
                    _ => unreachable!(),
                };

                found.push(TypeParameterAdded { defaulted: new_default || is_fn });
                id_mapping.add_type_param(new_type);
                id_mapping.add_non_mapped(new_type.def_id);
            },
            (None, None) => unreachable!(),
        }
    }

    for change_type in found.drain(..) {
        changes.add_change(change_type, old, None);
    }
}

// Below functions constitute the third pass of analysis, in which the types and/or trait bounds
// of matching items are compared for changes.

/// Given two items, compare their types.
fn diff_types<'a, 'tcx>(changes: &mut ChangeSet<'tcx>,
                        id_mapping: &IdMapping,
                        tcx: TyCtxt<'a, 'tcx, 'tcx>,
                        old: Def,
                        new: Def) {
    use rustc::hir::def::Def::*;

    let old_def_id = old.def_id();
    let new_def_id = new.def_id();

    // bail out of analysis of already broken items
    if changes.item_breaking(old_def_id) ||
            id_mapping.get_trait_def(&old_def_id)
                .map_or(false, |did| changes.trait_item_breaking(did)) {
        return;
    }

    match old {
        // type aliases, consts and statics just need their type to be checked
        TyAlias(_) | Const(_) | Static(_, _) => {
            cmp_types(changes,
                      id_mapping,
                      tcx,
                      old_def_id,
                      new_def_id,
                      tcx.type_of(old_def_id),
                      tcx.type_of(new_def_id));
        },
        // functions and methods require us to compare their signatures, not types
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
        // ADTs' types are compared field-wise
        Struct(_) | Enum(_) | Union(_) => {
            if let Some(children) = id_mapping.children_of(old_def_id) {
                for (o_def_id, n_def_id) in children {
                    let o_ty = tcx.type_of(o_def_id);
                    let n_ty = tcx.type_of(n_def_id);

                    cmp_types(changes, id_mapping, tcx, old_def_id, new_def_id, o_ty, n_ty);
                }
            }
        },
        // a trait definition has no type, so only it's trait bounds are compared
        Trait(_) => {
            cmp_bounds(changes, id_mapping, tcx, old_def_id, new_def_id);
        },
        _ => (),
    }
}

/// Compare two types and their trait bounds, possibly registering the resulting change.
fn cmp_types<'a, 'tcx>(changes: &mut ChangeSet<'tcx>,
                       id_mapping: &IdMapping,
                       tcx: TyCtxt<'a, 'tcx, 'tcx>,
                       orig_def_id: DefId,
                       target_def_id: DefId,
                       orig: Ty<'tcx>,
                       target: Ty<'tcx>) {
    info!("comparing types and bounds of {:?} / {:?}:\n  {:?} / {:?}",
          orig_def_id, target_def_id, orig, target);

    tcx.infer_ctxt().enter(|infcx| {
        let compcx = TypeComparisonContext::target_new(&infcx, id_mapping, false);

        let orig_substs = Substs::identity_for_item(infcx.tcx, target_def_id);
        let orig = compcx.forward_trans.translate_item_type(orig_def_id, orig);
        // let orig = orig.subst(infcx.tcx, orig_substs);

        // functions need inference, other items can declare defaulted type parameters
        let target_substs = if target.is_fn() {
            compcx.compute_target_infer_substs(target_def_id)
        } else {
            compcx.compute_target_default_substs(target_def_id)
        };
        let target = target.subst(infcx.tcx, target_substs);

        let target_param_env =
            infcx.tcx.param_env(target_def_id).subst(infcx.tcx, target_substs);

        if let Some(err) =
            compcx.check_type_error(tcx, target_def_id, target_param_env, orig, target)
        {
            changes.add_change(TypeChanged { error: err }, orig_def_id, None);
        } else {
            // check the bounds if no type error has been found
            compcx.check_bounds_bidirectional(changes,
                                              tcx,
                                              orig_def_id,
                                              target_def_id,
                                              orig_substs,
                                              target_substs);
        }
    });
}

/// Compare the trait bounds of two items, possibly registering the resulting change.
fn cmp_bounds<'a, 'tcx>(changes: &mut ChangeSet<'tcx>,
                        id_mapping: &IdMapping,
                        tcx: TyCtxt<'a, 'tcx, 'tcx>,
                        orig_def_id: DefId,
                        target_def_id: DefId) {
    info!("comparing bounds of {:?} / {:?}", orig_def_id, target_def_id);

    tcx.infer_ctxt().enter(|infcx| {
        let compcx = TypeComparisonContext::target_new(&infcx, id_mapping, true);

        let orig_substs = Substs::identity_for_item(infcx.tcx, target_def_id);
        let target_substs = compcx.compute_target_default_substs(target_def_id);

        compcx.check_bounds_bidirectional(changes,
                                          tcx,
                                          orig_def_id,
                                          target_def_id,
                                          orig_substs,
                                          target_substs);
    })
}

// Below functions constitute the fourth pass of analysis, in which impls are matched up based on
// their trait bounds and compared for changes, if applicable.

/// Compare the inherent implementations of all matching items.
fn diff_inherent_impls<'a, 'tcx>(changes: &mut ChangeSet<'tcx>,
                                 id_mapping: &IdMapping,
                                 tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    debug!("diffing inherent impls");

    let to_new = TranslationContext::target_new(tcx, id_mapping, false);
    let to_old = TranslationContext::target_old(tcx, id_mapping, false);

    for (orig_item, orig_impls) in id_mapping.inherent_impls() {
        // determine where the item comes from
        let (forward_trans, err_type) =
            if id_mapping.in_old_crate(orig_item.parent_def_id) {
                (&to_new, AssociatedItemRemoved)
            } else if id_mapping.in_new_crate(orig_item.parent_def_id) {
                (&to_old, AssociatedItemAdded)
            } else {
                unreachable!()
            };

        // determine item visibility
        let parent_output = changes.get_output(orig_item.parent_def_id);

        // for each original impl the item is present in, ...
        for &(orig_impl_def_id, orig_item_def_id) in orig_impls {
            let orig_assoc_item = tcx.associated_item(orig_item_def_id);

            let item_span = tcx.def_span(orig_item_def_id);
            changes.new_change(orig_item_def_id,
                               orig_item_def_id,
                               orig_item.name,
                               item_span,
                               item_span,
                               parent_output && orig_assoc_item.vis == Public);

            // ... determine the set of target impls that serve as candidates
            let target_impls = if let Some(impls) = forward_trans
                .translate_inherent_entry(orig_item)
                .and_then(|item| id_mapping.get_inherent_impls(&item))
            {
                impls
            } else {
                changes.add_change(err_type.clone(), orig_item_def_id, None);
                continue;
            };

            // if any of the candidates matches, the item is compatible across versions
            let match_found = target_impls
                .iter()
                .any(|&(target_impl_def_id, target_item_def_id)| {
                    let target_assoc_item = tcx.associated_item(target_item_def_id);

                    if parent_output && target_assoc_item.vis == Public {
                        changes.set_output(orig_item.parent_def_id);
                    }

                    match_inherent_impl(changes,
                                        id_mapping,
                                        tcx,
                                        orig_impl_def_id,
                                        target_impl_def_id,
                                        orig_assoc_item,
                                        target_assoc_item)
                });

            // otherwise, it has been essentially added/removed
            if !match_found {
                changes.add_change(err_type.clone(), orig_item_def_id, None);
            }
        }
    }
}

/// Compare the implementations of all matching traits.
fn diff_trait_impls<'a, 'tcx>(changes: &mut ChangeSet<'tcx>,
                              id_mapping: &IdMapping,
                              tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    debug!("diffing trait impls");

    use rustc::hir::def::Def;
    let to_new = TranslationContext::target_new(tcx, id_mapping, false);
    let to_old = TranslationContext::target_old(tcx, id_mapping, false);

    for old_impl_def_id in tcx.all_trait_implementations(id_mapping.get_old_crate()).iter() {
        let old_trait_def_id = tcx.impl_trait_ref(*old_impl_def_id).unwrap().def_id;

        let old_impl_parent_def =
            tcx.parent_def_id(*old_impl_def_id).and_then(|did| tcx.describe_def(did));
        let old_impl_parent_is_fn = match old_impl_parent_def {
            Some(Def::Fn(_)) | Some(Def::Method(_)) => true,
            _ => false,
        };

        if !to_new.can_translate(old_trait_def_id) || old_impl_parent_is_fn {
            continue;
        }

        if !match_trait_impl(tcx, &to_new, *old_impl_def_id) {
            changes.new_change_impl(*old_impl_def_id,
                                    tcx.item_path_str(*old_impl_def_id),
                                    tcx.def_span(*old_impl_def_id));
            changes.add_change(TraitImplTightened, *old_impl_def_id, None);
        }
    }

    for new_impl_def_id in tcx.all_trait_implementations(id_mapping.get_new_crate()).iter() {
        let new_trait_def_id = tcx.impl_trait_ref(*new_impl_def_id).unwrap().def_id;

        let new_impl_parent_def =
            tcx.parent_def_id(*new_impl_def_id).and_then(|did| tcx.describe_def(did));
        let new_impl_parent_is_fn = match new_impl_parent_def {
            Some(Def::Fn(_)) | Some(Def::Method(_)) => true,
            _ => false,
        };

        if !to_old.can_translate(new_trait_def_id) || new_impl_parent_is_fn {
            continue;
        }

        if !match_trait_impl(tcx, &to_old, *new_impl_def_id) {
            changes.new_change_impl(*new_impl_def_id,
                                    tcx.item_path_str(*new_impl_def_id),
                                    tcx.def_span(*new_impl_def_id));
            changes.add_change(TraitImplLoosened, *new_impl_def_id, None);
        }
    }
}

/// Compare an item pair in two inherent implementations and indicate whether the target one is
/// compatible with the original one.
fn match_inherent_impl<'a, 'tcx>(changes: &mut ChangeSet<'tcx>,
                                 id_mapping: &IdMapping,
                                 tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                 orig_impl_def_id: DefId,
                                 target_impl_def_id: DefId,
                                 orig_item: AssociatedItem,
                                 target_item: AssociatedItem) -> bool {
    use rustc::ty::AssociatedKind;

    debug!("match_inherent_impl: orig_impl/item: {:?}/{:?}, target_impl/item: {:?}/{:?}",
           orig_impl_def_id, orig_item, target_impl_def_id, target_item);

    let orig_item_def_id = orig_item.def_id;
    let target_item_def_id = target_item.def_id;

    tcx.infer_ctxt().enter(|infcx| {
        let (compcx, register_errors) = if id_mapping.in_old_crate(orig_impl_def_id) {
            (TypeComparisonContext::target_new(&infcx, id_mapping, false), true)
        } else {
            (TypeComparisonContext::target_old(&infcx, id_mapping, false), false)
        };

        let orig_substs = Substs::identity_for_item(infcx.tcx, target_item_def_id);
        let orig_self = compcx
            .forward_trans
            .translate_item_type(orig_impl_def_id, infcx.tcx.type_of(orig_impl_def_id));

        let target_substs = compcx.compute_target_infer_substs(target_item_def_id);
        let target_self = infcx.tcx.type_of(target_impl_def_id).subst(infcx.tcx, target_substs);

        let target_param_env = infcx.tcx.param_env(target_impl_def_id);

        let error = compcx.check_type_error(tcx,
                                            target_impl_def_id,
                                            target_param_env,
                                            orig_self,
                                            target_self);

        if error.is_some() {
            // `Self` on the impls isn't equal - no impl match.
            return false;
        }

        let orig_param_env = compcx
            .forward_trans
            .translate_param_env(orig_impl_def_id, tcx.param_env(orig_impl_def_id));

        if let Some(orig_param_env) = orig_param_env {
            let errors = compcx.check_bounds_error(tcx,
                                                   orig_param_env,
                                                   target_impl_def_id,
                                                   target_substs);
            if errors.is_some() {
                // The bounds on the impls have been tightened - no impl match.
                return false
            }
        } else {
            // The bounds could not have been translated - no impl match.
            return false;
        }

        // at this point we have an impl match, so the return value is always `true`.

        if !register_errors {
            // checking backwards, impls match.
            return true;
        }

        // prepare the item type for comparison, as we do for toplevel items' types
        let (orig, target) = match (orig_item.kind, target_item.kind) {
            (AssociatedKind::Const, AssociatedKind::Const) |
            (AssociatedKind::Type, AssociatedKind::Type) => {
                (infcx.tcx.type_of(orig_item_def_id), infcx.tcx.type_of(target_item_def_id))
            },
            (AssociatedKind::Method, AssociatedKind::Method) => {
                diff_method(changes, tcx, orig_item, target_item);
                let orig_sig = infcx.tcx.type_of(orig_item_def_id).fn_sig(tcx);
                let target_sig = infcx.tcx.type_of(target_item_def_id).fn_sig(tcx);
                (tcx.mk_fn_ptr(orig_sig), tcx.mk_fn_ptr(target_sig))
            },
            _ => {
                unreachable!()
            },
        };

        let orig = compcx.forward_trans.translate_item_type(orig_item_def_id, orig);
        let target = target.subst(infcx.tcx, target_substs);

        let error = compcx.check_type_error(tcx,
                                            target_item_def_id,
                                            target_param_env,
                                            orig,
                                            target);

        if let Some(err) = error {
            changes.add_change(TypeChanged { error: err }, orig_item_def_id, None);
        } else {
            // check the bounds if no type error has been found
            compcx.check_bounds_bidirectional(changes,
                                              tcx,
                                              orig_item_def_id,
                                              target_item_def_id,
                                              orig_substs,
                                              target_substs);
        }

        true
    })
}

/// Compare two implementations and indicate whether the target one is compatible with the
/// original one.
fn match_trait_impl<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              trans: &TranslationContext<'a, 'tcx, 'tcx>,
                              orig_def_id: DefId) -> bool {
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
