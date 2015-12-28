// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Logic and data structures related to impl specialization, explained in
// greater detail below.
//
// At the moment, this implementation support only the simple "chain" rule:
// If any two impls overlap, one must be a strict subset of the other.

use super::util;
use super::SelectionContext;

use middle::cstore::CrateStore;
use middle::def_id::DefId;
use middle::infer::{self, InferCtxt, TypeOrigin};
use middle::region;
use middle::subst::{Subst, Substs};
use middle::traits;
use middle::ty::{self, ImplOrTraitItem};
use syntax::codemap::DUMMY_SP;
use util::nodemap::DefIdMap;

/// A per-trait graph of impls in specialization order.
///
/// The graph provides two key services:
///
/// - Construction, which implicitly checks for overlapping impls (i.e., impls
///   that overlap but where neither specializes the other -- an artifact of the
///   simple "chain" rule.
///
/// - Parent extraction. In particular, the graph can give you the *immediate*
///   parents of a given specializing impl, which is needed for extracting
///   default items amongst other thigns. In the simple "chain" rule, every impl
///   has at most one parent.
pub struct SpecializationGraph {
    // all impls have a parent; the "root" impls have as their parent the def_id
    // of the trait
    parent: DefIdMap<DefId>,

    // the "root" impls are found by looking up the trait's def_id.
    children: DefIdMap<Vec<DefId>>,
}

/// Information pertinent to an overlapping impl error.
pub struct Overlap<'tcx> {
    pub with_impl: DefId,

    /// NB: this TraitRef can contain inference variables!
    pub on_trait_ref: ty::TraitRef<'tcx>,
}

impl SpecializationGraph {
    pub fn new() -> SpecializationGraph {
        SpecializationGraph {
            parent: Default::default(),
            children: Default::default(),
        }
    }

    /// Insert a local impl into the specialization graph. If an existing impl
    /// conflicts with it (has overlap, but neither specializes the other),
    /// information about the area of overlap is returned in the `Err`.
    pub fn insert<'tcx>(&mut self,
                        tcx: &ty::ctxt<'tcx>,
                        impl_def_id: DefId,
                        trait_ref: ty::TraitRef)
                        -> Result<(), Overlap<'tcx>> {
        assert!(impl_def_id.is_local());

        let mut parent = trait_ref.def_id;
        let mut my_children = vec![];

        // descend the existing tree, looking for the right location to add this impl
        'descend: loop {
            let mut possible_siblings = self.children.entry(parent).or_insert(vec![]);

            for slot in possible_siblings.iter_mut() {
                let possible_sibling = *slot;

                let infcx = infer::new_infer_ctxt(tcx, &tcx.tables, None, false);
                let overlap = traits::overlapping_impls(&infcx, possible_sibling, impl_def_id);

                if let Some(trait_ref) = overlap {
                    let le = specializes(tcx, impl_def_id, possible_sibling);
                    let ge = specializes(tcx, possible_sibling, impl_def_id);

                    if le && !ge {
                        // the impl specializes possible_sibling
                        parent = possible_sibling;
                        continue 'descend;
                    } else if ge && !le {
                        // possible_sibling specializes the impl
                        *slot = impl_def_id;
                        self.parent.insert(possible_sibling, impl_def_id);
                        my_children.push(possible_sibling);
                    } else {
                        // overlap, but no specialization; error out
                        return Err(Overlap {
                            with_impl: possible_sibling,
                            on_trait_ref: trait_ref,
                        });
                    }

                    break 'descend;
                }
            }

            // no overlap with any potential siblings, so add as a new sibling
            self.parent.insert(impl_def_id, parent);
            possible_siblings.push(impl_def_id);
            break;
        }

        if self.children.insert(impl_def_id, my_children).is_some() {
            tcx.sess
               .bug("When inserting an impl into the specialization graph, existing children for \
                     the impl were already present.");
        }

        Ok(())
    }

    /// Insert cached metadata mapping from a child impl back to its parent.
    pub fn record_impl_from_cstore(&mut self, parent: DefId, child: DefId) {
        if self.parent.insert(child, parent).is_some() {
            panic!("When recording an impl from the crate store, information about its parent \
                    was already present.");
        }

        self.children.entry(parent).or_insert(vec![]).push(child);
    }

    /// The parent of a given impl, which is the def id of the trait when the
    /// impl is a "specialization root".
    pub fn parent(&self, child: DefId) -> DefId {
        *self.parent.get(&child).unwrap()
    }
}

/// When we have selected one impl, but are actually using item definitions from
/// a parent impl providing a default, we need a way to translate between the
/// type parameters of the two impls. Here the `source_impl` is the one we've
/// selected, and `source_substs` is a substitution of its generics (and possibly
/// some relevant `FnSpace` variables as well). And `target_impl` is the impl
/// we're actually going to get the definition from.
fn translate_substs_between_impls<'tcx>(tcx: &ty::ctxt<'tcx>,
                                        source_impl: DefId,
                                        source_substs: Substs<'tcx>,
                                        target_impl: DefId)
                                        -> Substs<'tcx> {

    // We need to build a subst that covers all the generics of
    // `target_impl`. Start by introducing fresh infer variables:
    let target_generics = tcx.lookup_item_type(target_impl).generics;
    let mut infcx = infer::normalizing_infer_ctxt(tcx, &tcx.tables);
    let mut target_substs = infcx.fresh_substs_for_generics(DUMMY_SP, &target_generics);
    if source_substs.regions.is_erased() {
        target_substs = target_substs.erase_regions()
    }

    if !fulfill_implication(&mut infcx,
                            source_impl,
                            source_substs.clone(),
                            target_impl,
                            target_substs.clone()) {
        tcx.sess
           .bug("When translating substitutions for specialization, the expected specializaiton \
                 failed to hold")
    }

    // Now resolve the *substitution* we built for the target earlier, replacing
    // the inference variables inside with whatever we got from fulfillment. We
    // also carry along any FnSpace substitutions, which don't need to be
    // adjusted when mapping from one impl to another.
    infcx.resolve_type_vars_if_possible(&target_substs)
         .with_method_from_subst(&source_substs)
}

/// When we've selected an impl but need to use an item definition provided by
/// the trait itself, we need to translate the substitution applied to the impl
/// to one that makes sense for the trait.
fn translate_substs_from_impl_to_trait<'tcx>(tcx: &ty::ctxt<'tcx>,
                                             source_impl: DefId,
                                             source_substs: Substs<'tcx>)
                                             -> Substs<'tcx> {

    let source_trait_ref = tcx.impl_trait_ref(source_impl).unwrap().subst(tcx, &source_substs);

    let mut new_substs = source_trait_ref.substs.clone();
    if source_substs.regions.is_erased() {
        new_substs = new_substs.erase_regions()
    }

    // Carry any FnSpace substitutions along; they don't need to be adjusted
    new_substs.with_method_from_subst(&source_substs)
}

#[derive(Debug, Copy, Clone)]
/// When looking up an item in an impl, it may turn out that the item
/// is actually provided as a default by a more generic impl, or by
/// the trait itself. This enum says where the item came from.
pub enum ItemSource {
    Impl {
        requested_impl: DefId,
        actual_impl: DefId,
    },
    Trait {
        requested_impl: DefId,
    },
}

impl ItemSource {
    pub fn is_from_trait(&self) -> bool {
        match *self {
            ItemSource::Trait { .. } => true,
            _ => false,
        }
    }

    /// Given a subst for the requested impl, translate it to a subst
    /// appropriate for the actual item definition (whether it be in that impl,
    /// a parent impl, or the trait).
    pub fn translate_substs<'tcx>(&self,
                                  tcx: &ty::ctxt<'tcx>,
                                  requested_impl_substs: Substs<'tcx>)
                                  -> Substs<'tcx> {
        match *self {
            ItemSource::Impl { requested_impl, actual_impl } => {
                // no need to translate if we're targetting the impl we started with
                if requested_impl == actual_impl {
                    return requested_impl_substs;
                }

                translate_substs_between_impls(tcx,
                                               requested_impl,
                                               requested_impl_substs,
                                               actual_impl)

            }
            ItemSource::Trait { requested_impl } => {
                translate_substs_from_impl_to_trait(tcx, requested_impl, requested_impl_substs)
            }
        }
    }
}

/// Lookup the definition of an item within `requested_impl` or its specialization
/// parents, including provided items from the trait itself.
///
/// The closure `f` works in the style of `filter_map`.
pub fn get_impl_item_or_default<'tcx, I, F>(tcx: &ty::ctxt<'tcx>,
                                            requested_impl: DefId,
                                            mut f: F)
                                            -> Option<(I, ItemSource)>
    where F: for<'a> FnMut(&ImplOrTraitItem<'tcx>) -> Option<I>
{
    let impl_or_trait_items_map = tcx.impl_or_trait_items.borrow();
    let trait_def_id = tcx.trait_id_of_impl(requested_impl).unwrap();
    let trait_def = tcx.lookup_trait_def(trait_def_id);

    // Walk up the specialization tree, looking for a matching item definition

    let mut current_impl = requested_impl;
    loop {
        for impl_item_id in &tcx.impl_items.borrow()[&current_impl] {
            let impl_item = &impl_or_trait_items_map[&impl_item_id.def_id()];
            if let Some(t) = f(impl_item) {
                let source = ItemSource::Impl {
                    requested_impl: requested_impl,
                    actual_impl: current_impl,
                };
                return Some((t, source));
            }
        }

        if let Some(parent) = trait_def.parent_of_impl(current_impl) {
            current_impl = parent;
        } else {
            break;
        }
    }

    // The item isn't defined anywhere in the hierarchy. Get the
    // default from the trait.

    for trait_item in tcx.trait_items(trait_def_id).iter() {
        if let Some(t) = f(trait_item) {
            return Some((t, ItemSource::Trait { requested_impl: requested_impl }));
        }
    }

    None
}

fn skolemizing_subst_for_impl<'a>(tcx: &ty::ctxt<'a>, impl_def_id: DefId) -> Substs<'a> {
    let impl_generics = tcx.lookup_item_type(impl_def_id).generics;

    let types = impl_generics.types.map(|def| tcx.mk_param_from_def(def));

    // FIXME: figure out what we actually want here
    let regions = impl_generics.regions.map(|_| ty::Region::ReStatic);
    // |d| infcx.next_region_var(infer::RegionVariableOrigin::EarlyBoundRegion(span, d.name)));

    Substs::new(types, regions)
}

/// Is impl1 a specialization of impl2?
///
/// Specialization is determined by the sets of types to which the impls apply;
/// impl1 specializes impl2 if it applies to a subset of the types impl2 applies
/// to.
pub fn specializes(tcx: &ty::ctxt, impl1_def_id: DefId, impl2_def_id: DefId) -> bool {
    // We determine whether there's a subset relationship by:
    //
    // - skolemizing impl1,
    // - instantiating impl2 with fresh inference variables,
    // - assuming the where clauses for impl1,
    // - unifying,
    // - attempting to prove the where clauses for impl2
    //
    // The last three steps are essentially checking for an implication between two impls
    // after appropriate substitutions. This is what `fulfill_implication` checks for.
    //
    // See RFC 1210 for more details and justification.

    let mut infcx = infer::normalizing_infer_ctxt(tcx, &tcx.tables);

    let impl1_substs = skolemizing_subst_for_impl(tcx, impl1_def_id);
    let impl2_substs = util::fresh_type_vars_for_impl(&infcx, DUMMY_SP, impl2_def_id);

    fulfill_implication(&mut infcx,
                        impl1_def_id,
                        impl1_substs,
                        impl2_def_id,
                        impl2_substs)
}

/// Does impl1 (instantiated with the impl1_substs) imply impl2
/// (instantiated with impl2_substs)?
///
/// Mutates the `infcx` in two ways:
/// - by adding the obligations of impl1 to the parameter environment
/// - via fulfillment, so that if the implication holds the various unifications
fn fulfill_implication<'a, 'tcx>(infcx: &mut InferCtxt<'a, 'tcx>,
                                 impl1_def_id: DefId,
                                 impl1_substs: Substs<'tcx>,
                                 impl2_def_id: DefId,
                                 impl2_substs: Substs<'tcx>)
                                 -> bool {
    let tcx = &infcx.tcx;

    let (impl1_trait_ref, impl1_obligations) = {
        let selcx = &mut SelectionContext::new(&infcx);
        util::impl_trait_ref_and_oblig(selcx, impl1_def_id, &impl1_substs)
    };

    let impl1_predicates: Vec<_> = impl1_obligations.iter()
                                                    .cloned()
                                                    .map(|oblig| oblig.predicate)
                                                    .collect();

    infcx.parameter_environment = ty::ParameterEnvironment {
        tcx: tcx,
        free_substs: impl1_substs,
        implicit_region_bound: ty::ReEmpty, // FIXME: is this OK?
        caller_bounds: impl1_predicates,
        selection_cache: traits::SelectionCache::new(),
        evaluation_cache: traits::EvaluationCache::new(),
        free_id_outlive: region::DUMMY_CODE_EXTENT, // FIXME: is this OK?
    };

    let selcx = &mut SelectionContext::new(&infcx);
    let (impl2_trait_ref, impl2_obligations) = util::impl_trait_ref_and_oblig(selcx,
                                                                              impl2_def_id,
                                                                              &impl2_substs);

    // do the impls unify? If not, no specialization.
    if let Err(_) = infer::mk_eq_trait_refs(&infcx,
                                            true,
                                            TypeOrigin::Misc(DUMMY_SP),
                                            impl1_trait_ref,
                                            impl2_trait_ref) {
        debug!("fulfill_implication: {:?} does not unify with {:?}",
               impl1_trait_ref,
               impl2_trait_ref);
        return false;
    }

    let mut fulfill_cx = infcx.fulfillment_cx.borrow_mut();

    // attempt to prove all of the predicates for impl2 given those for impl1
    // (which are packed up in penv)

    for oblig in impl2_obligations.into_iter() {
        fulfill_cx.register_predicate_obligation(&infcx, oblig);
    }

    if let Err(errors) = infer::drain_fulfillment_cx(&infcx, &mut fulfill_cx, &()) {
        // no dice!
        debug!("fulfill_implication: for impls on {:?} and {:?}, could not fulfill: {:?} given \
                {:?}",
               impl1_trait_ref,
               impl2_trait_ref,
               errors,
               infcx.parameter_environment.caller_bounds);
        false
    } else {
        debug!("fulfill_implication: an impl for {:?} specializes {:?} (`where` clauses elided)",
               impl1_trait_ref,
               impl2_trait_ref);
        true
    }
}
