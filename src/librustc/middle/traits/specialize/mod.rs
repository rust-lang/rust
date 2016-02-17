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
//
// See traits/README.md for a bit more detail on how specialization
// fits together with the rest of the trait machinery.

use super::{util, build_selcx, SelectionContext};

use middle::cstore::CrateStore;
use middle::def_id::DefId;
use middle::infer::{self, InferCtxt, TypeOrigin};
use middle::region;
use middle::subst::{Subst, Substs};
use middle::traits;
use middle::ty;
use syntax::codemap::DUMMY_SP;

pub mod specialization_graph;

/// Information pertinent to an overlapping impl error.
pub struct Overlap<'a, 'tcx: 'a> {
    pub in_context: InferCtxt<'a, 'tcx>,
    pub with_impl: DefId,
    pub on_trait_ref: ty::TraitRef<'tcx>,
}

/// Given a subst for the requested impl, translate it to a subst
/// appropriate for the actual item definition (whether it be in that impl,
/// a parent impl, or the trait).
pub fn translate_substs<'tcx>(tcx: &ty::ctxt<'tcx>,
                              from_impl: DefId,
                              from_impl_substs: Substs<'tcx>,
                              to_node: specialization_graph::Node)
                              -> Substs<'tcx> {
    match to_node {
        specialization_graph::Node::Impl(to_impl) => {
            // no need to translate if we're targetting the impl we started with
            if from_impl == to_impl {
                return from_impl_substs;
            }

            translate_substs_between_impls(tcx, from_impl, from_impl_substs, to_impl)

        }
        specialization_graph::Node::Trait(..) => {
            translate_substs_from_impl_to_trait(tcx, from_impl, from_impl_substs)
        }
    }
}

/// When we have selected one impl, but are actually using item definitions from
/// a parent impl providing a default, we need a way to translate between the
/// type parameters of the two impls. Here the `source_impl` is the one we've
/// selected, and `source_substs` is a substitution of its generics (and
/// possibly some relevant `FnSpace` variables as well). And `target_impl` is
/// the impl we're actually going to get the definition from. The resulting
/// substitution will map from `target_impl`'s generics to `source_impl`'s
/// generics as instantiated by `source_subst`.
///
/// For example, consider the following scenario:
///
/// ```rust
/// trait Foo { ... }
/// impl<T, U> Foo for (T, U) { ... }  // target impl
/// impl<V> Foo for (V, V) { ... }     // source impl
/// ```
///
/// Suppose we have selected "source impl" with `V` instantiated with `u32`.
/// This function will produce a substitution with `T` and `U` both mapping to `u32`.
///
/// Where clauses add some trickiness here, because they can be used to "define"
/// an argument indirectly:
///
/// ```rust
/// impl<'a, I, T: 'a> Iterator for Cloned<I>
///    where I: Iterator<Item=&'a T>, T: Clone
/// ```
///
/// In a case like this, the substitution for `T` is determined indirectly,
/// through associated type projection. We deal with such cases by using
/// *fulfillment* to relate the two impls, requiring that all projections are
/// resolved.
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
    if !tcx.sess.features.borrow().specialization {
        return false;
    }

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

    let selcx = &mut build_selcx(&infcx).project_topmost().build();
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
