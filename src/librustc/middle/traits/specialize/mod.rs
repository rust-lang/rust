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

use super::{SelectionContext, FulfillmentContext};
use super::util::{fresh_type_vars_for_impl, impl_trait_ref_and_oblig};

use middle::cstore::CrateStore;
use middle::def_id::DefId;
use middle::infer::{self, InferCtxt, TypeOrigin};
use middle::region;
use middle::subst::{Subst, Substs};
use middle::traits::{self, ProjectionMode};
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
//
// When we have selected one impl, but are actually using item definitions from
// a parent impl providing a default, we need a way to translate between the
// type parameters of the two impls. Here the `source_impl` is the one we've
// selected, and `source_substs` is a substitution of its generics (and
// possibly some relevant `FnSpace` variables as well). And `target_node` is
// the impl/trait we're actually going to get the definition from. The resulting
// substitution will map from `target_node`'s generics to `source_impl`'s
// generics as instantiated by `source_subst`.
//
// For example, consider the following scenario:
//
// ```rust
// trait Foo { ... }
// impl<T, U> Foo for (T, U) { ... }  // target impl
// impl<V> Foo for (V, V) { ... }     // source impl
// ```
//
// Suppose we have selected "source impl" with `V` instantiated with `u32`.
// This function will produce a substitution with `T` and `U` both mapping to `u32`.
//
// Where clauses add some trickiness here, because they can be used to "define"
// an argument indirectly:
//
// ```rust
// impl<'a, I, T: 'a> Iterator for Cloned<I>
//    where I: Iterator<Item=&'a T>, T: Clone
// ```
//
// In a case like this, the substitution for `T` is determined indirectly,
// through associated type projection. We deal with such cases by using
// *fulfillment* to relate the two impls, requiring that all projections are
// resolved.
pub fn translate_substs<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                  source_impl: DefId,
                                  source_substs: Substs<'tcx>,
                                  target_node: specialization_graph::Node)
                                  -> Substs<'tcx>
{
    let source_trait_ref = infcx.tcx
                                .impl_trait_ref(source_impl)
                                .unwrap()
                                .subst(infcx.tcx, &source_substs);

    // translate the Self and TyParam parts of the substitution, since those
    // vary across impls
    let target_substs = match target_node {
        specialization_graph::Node::Impl(target_impl) => {
            // no need to translate if we're targetting the impl we started with
            if source_impl == target_impl {
                return source_substs;
            }

            fulfill_implication(infcx, source_trait_ref, target_impl).unwrap_or_else(|_| {
                infcx.tcx
                     .sess
                     .bug("When translating substitutions for specialization, the expected \
                           specializaiton failed to hold")
            })
        }
        specialization_graph::Node::Trait(..) => source_trait_ref.substs.clone(),
    };

    // retain erasure mode
    // NB: this must happen before inheriting method generics below
    let target_substs = if source_substs.regions.is_erased() {
        target_substs.erase_regions()
    } else {
        target_substs
    };

    // directly inherent the method generics, since those do not vary across impls
    target_substs.with_method_from_subst(&source_substs)
}


fn skolemizing_subst_for_impl<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                        impl_def_id: DefId)
                                        -> Substs<'tcx>
{
    let impl_generics = infcx.tcx.lookup_item_type(impl_def_id).generics;

    let types = impl_generics.types.map(|def| infcx.tcx.mk_param_from_def(def));

    // TODO: figure out what we actually want here
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
    // - assuming the where clauses for impl1,
    // - instantiating impl2 with fresh inference variables,
    // - unifying,
    // - attempting to prove the where clauses for impl2
    //
    // The last three steps are encapsulated in `fulfill_implication`.
    //
    // See RFC 1210 for more details and justification.

    // Currently we do not allow e.g. a negative impl to specialize a positive one
    if tcx.trait_impl_polarity(impl1_def_id) != tcx.trait_impl_polarity(impl2_def_id) {
        return false
    }

    let mut infcx = infer::normalizing_infer_ctxt(tcx, &tcx.tables, ProjectionMode::Topmost);

    // Skiolemize impl1: we want to prove that "for all types matched by impl1,
    // those types are also matched by impl2".
    let impl1_substs = skolemizing_subst_for_impl(&infcx, impl1_def_id);
    let (impl1_trait_ref, impl1_obligations) = {
        let selcx = &mut SelectionContext::new(&infcx);
        impl_trait_ref_and_oblig(selcx, impl1_def_id, &impl1_substs)
    };

    // Add impl1's obligations as assumptions to the environment.
    let impl1_predicates: Vec<_> = impl1_obligations.iter()
                                                    .cloned()
                                                    .map(|oblig| oblig.predicate)
                                                    .collect();
    infcx.parameter_environment = ty::ParameterEnvironment {
        tcx: tcx,
        free_substs: impl1_substs,
        implicit_region_bound: ty::ReEmpty, // TODO: is this OK?
        caller_bounds: impl1_predicates,
        selection_cache: traits::SelectionCache::new(),
        evaluation_cache: traits::EvaluationCache::new(),
        free_id_outlive: region::DUMMY_CODE_EXTENT, // TODO: is this OK?
    };

    // Attempt to prove that impl2 applies, given all of the above.
    fulfill_implication(&infcx, impl1_trait_ref, impl2_def_id).is_ok()
}

/// Attempt to fulfill all obligations of `target_impl` after unification with
/// `source_trait_ref`. If successful, returns a substitution for *all* the
/// generics of `target_impl`, including both those needed to unify with
/// `source_trait_ref` and those whose identity is determined via a where
/// clause in the impl.
fn fulfill_implication<'a, 'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                 source_trait_ref: ty::TraitRef<'tcx>,
                                 target_impl: DefId)
                                 -> Result<Substs<'tcx>, ()>
{
    infcx.probe(|_| {
        let selcx = &mut SelectionContext::new(&infcx);
        let target_substs = fresh_type_vars_for_impl(&infcx, DUMMY_SP, target_impl);
        let (target_trait_ref, obligations) = impl_trait_ref_and_oblig(selcx,
                                                                       target_impl,
                                                                       &target_substs);

        // do the impls unify? If not, no specialization.
        if let Err(_) = infer::mk_eq_trait_refs(&infcx,
                                                true,
                                                TypeOrigin::Misc(DUMMY_SP),
                                                source_trait_ref,
                                                target_trait_ref) {
            debug!("fulfill_implication: {:?} does not unify with {:?}",
                   source_trait_ref,
                   target_trait_ref);
            return Err(());
        }

        // attempt to prove all of the predicates for impl2 given those for impl1
        // (which are packed up in penv)

        let mut fulfill_cx = FulfillmentContext::new();
        for oblig in obligations.into_iter() {
            fulfill_cx.register_predicate_obligation(&infcx, oblig);
        }

        if let Err(errors) = infer::drain_fulfillment_cx(&infcx, &mut fulfill_cx, &()) {
            // no dice!
            debug!("fulfill_implication: for impls on {:?} and {:?}, could not fulfill: {:?} \
                    given {:?}",
                   source_trait_ref,
                   target_trait_ref,
                   errors,
                   infcx.parameter_environment.caller_bounds);
            Err(())
        } else {
            debug!("fulfill_implication: an impl for {:?} specializes {:?} (`where` clauses \
                    elided)",
                   source_trait_ref,
                   target_trait_ref);

            // Now resolve the *substitution* we built for the target earlier, replacing
            // the inference variables inside with whatever we got from fulfillment.

            // TODO: should this use `fully_resolve` instead?
            Ok(infcx.resolve_type_vars_if_possible(&target_substs))
        }
    })
}
