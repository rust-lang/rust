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

use rustc_data_structures::fnv::FnvHashMap;
use hir::def_id::DefId;
use infer::{InferCtxt, TypeOrigin};
use middle::region;
use ty::subst::{Subst, Substs};
use traits::{self, ProjectionMode, ObligationCause, Normalized};
use ty::{self, TyCtxt};
use syntax_pos::DUMMY_SP;

pub mod specialization_graph;

/// Information pertinent to an overlapping impl error.
pub struct OverlapError {
    pub with_impl: DefId,
    pub trait_desc: String,
    pub self_desc: Option<String>
}

/// Given a subst for the requested impl, translate it to a subst
/// appropriate for the actual item definition (whether it be in that impl,
/// a parent impl, or the trait).
/// When we have selected one impl, but are actually using item definitions from
/// a parent impl providing a default, we need a way to translate between the
/// type parameters of the two impls. Here the `source_impl` is the one we've
/// selected, and `source_substs` is a substitution of its generics (and
/// possibly some relevant `FnSpace` variables as well). And `target_node` is
/// the impl/trait we're actually going to get the definition from. The resulting
/// substitution will map from `target_node`'s generics to `source_impl`'s
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
pub fn translate_substs<'a, 'gcx, 'tcx>(infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                                        source_impl: DefId,
                                        source_substs: &'tcx Substs<'tcx>,
                                        target_node: specialization_graph::Node)
                                        -> &'tcx Substs<'tcx> {
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
                bug!("When translating substitutions for specialization, the expected \
                      specializaiton failed to hold")
            })
        }
        specialization_graph::Node::Trait(..) => source_trait_ref.substs,
    };

    // directly inherent the method generics, since those do not vary across impls
    infcx.tcx.mk_substs(target_substs.with_method_from_subst(source_substs))
}

/// Is impl1 a specialization of impl2?
///
/// Specialization is determined by the sets of types to which the impls apply;
/// impl1 specializes impl2 if it applies to a subset of the types impl2 applies
/// to.
pub fn specializes<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             impl1_def_id: DefId,
                             impl2_def_id: DefId) -> bool {
    if let Some(r) = tcx.specializes_cache.borrow().check(impl1_def_id, impl2_def_id) {
        return r;
    }

    // The feature gate should prevent introducing new specializations, but not
    // taking advantage of upstream ones.
    if !tcx.sess.features.borrow().specialization &&
        (impl1_def_id.is_local() || impl2_def_id.is_local()) {
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
        return false;
    }

    // create a parameter environment corresponding to a (skolemized) instantiation of impl1
    let scheme = tcx.lookup_item_type(impl1_def_id);
    let predicates = tcx.lookup_predicates(impl1_def_id);
    let mut penv = tcx.construct_parameter_environment(DUMMY_SP,
                                                       &scheme.generics,
                                                       &predicates,
                                                       region::DUMMY_CODE_EXTENT);
    let impl1_trait_ref = tcx.impl_trait_ref(impl1_def_id)
                             .unwrap()
                             .subst(tcx, &penv.free_substs);

    let result = tcx.normalizing_infer_ctxt(ProjectionMode::Topmost).enter(|mut infcx| {
        // Normalize the trait reference, adding any obligations
        // that arise into the impl1 assumptions.
        let Normalized { value: impl1_trait_ref, obligations: normalization_obligations } = {
            let selcx = &mut SelectionContext::new(&infcx);
            traits::normalize(selcx, ObligationCause::dummy(), &impl1_trait_ref)
        };
        penv.caller_bounds.extend(normalization_obligations.into_iter().map(|o| {
            match tcx.lift_to_global(&o.predicate) {
                Some(predicate) => predicate,
                None => {
                    bug!("specializes: obligation `{:?}` has inference types/regions", o);
                }
            }
        }));

        // Install the parameter environment, taking the predicates of impl1 as assumptions:
        infcx.parameter_environment = penv;

        // Attempt to prove that impl2 applies, given all of the above.
        fulfill_implication(&infcx, impl1_trait_ref, impl2_def_id).is_ok()
    });

    tcx.specializes_cache.borrow_mut().insert(impl1_def_id, impl2_def_id, result);
    result
}

/// Attempt to fulfill all obligations of `target_impl` after unification with
/// `source_trait_ref`. If successful, returns a substitution for *all* the
/// generics of `target_impl`, including both those needed to unify with
/// `source_trait_ref` and those whose identity is determined via a where
/// clause in the impl.
fn fulfill_implication<'a, 'gcx, 'tcx>(infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                                       source_trait_ref: ty::TraitRef<'tcx>,
                                       target_impl: DefId)
                                       -> Result<&'tcx Substs<'tcx>, ()> {
    let selcx = &mut SelectionContext::new(&infcx);
    let target_substs = fresh_type_vars_for_impl(&infcx, DUMMY_SP, target_impl);
    let (target_trait_ref, obligations) = impl_trait_ref_and_oblig(selcx,
                                                                   target_impl,
                                                                   &target_substs);

    // do the impls unify? If not, no specialization.
    if let Err(_) = infcx.eq_trait_refs(true,
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

    if let Err(errors) = infcx.drain_fulfillment_cx(&mut fulfill_cx, &()) {
        // no dice!
        debug!("fulfill_implication: for impls on {:?} and {:?}, could not fulfill: {:?} given \
                {:?}",
               source_trait_ref,
               target_trait_ref,
               errors,
               infcx.parameter_environment.caller_bounds);
        Err(())
    } else {
        debug!("fulfill_implication: an impl for {:?} specializes {:?}",
               source_trait_ref,
               target_trait_ref);

        // Now resolve the *substitution* we built for the target earlier, replacing
        // the inference variables inside with whatever we got from fulfillment.
        Ok(infcx.resolve_type_vars_if_possible(&target_substs))
    }
}

pub struct SpecializesCache {
    map: FnvHashMap<(DefId, DefId), bool>
}

impl SpecializesCache {
    pub fn new() -> Self {
        SpecializesCache {
            map: FnvHashMap()
        }
    }

    pub fn check(&self, a: DefId, b: DefId) -> Option<bool> {
        self.map.get(&(a, b)).cloned()
    }

    pub fn insert(&mut self, a: DefId, b: DefId, result: bool) {
        self.map.insert((a, b), result);
    }
}
