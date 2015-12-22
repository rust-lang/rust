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
use middle::ty;
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

        let infcx = infer::new_infer_ctxt(tcx, &tcx.tables, None, false);
        let mut parent = trait_ref.def_id;

        let mut my_children = vec![];

        // descend the existing tree, looking for the right location to add this impl
        'descend: loop {
            let mut possible_siblings = self.children.entry(parent).or_insert(vec![]);

            for slot in possible_siblings.iter_mut() {
                let possible_sibling = *slot;

                let overlap = infcx.probe(|_| {
                    traits::overlapping_impls(&infcx, possible_sibling, impl_def_id)
                });

                if let Some(trait_ref) = overlap {
                    let le = specializes(&infcx, impl_def_id, possible_sibling);
                    let ge = specializes(&infcx, possible_sibling, impl_def_id);

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
            panic!("When inserting an impl into the specialization graph, existing children for \
                    the impl were already present.");
        }

        Ok(())
    }

    /// Insert cached metadata mapping from a child impl back to its parent
    pub fn record_impl_from_cstore(&mut self, parent: DefId, child: DefId) {
        if self.parent.insert(child, Some(parent)).is_some() {
            panic!("When recording an impl from the crate store, information about its parent \
                    was already present.");
        }

        self.children.entry(parent).or_insert(vec![]).push(child);
    }
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
pub fn specializes(infcx: &InferCtxt, impl1_def_id: DefId, impl2_def_id: DefId) -> bool {
    let tcx = &infcx.tcx;

    // We determine whether there's a subset relationship by:
    //
    // - skolemizing impl1,
    // - assuming the where clauses for impl1,
    // - unifying,
    // - attempting to prove the where clauses for impl2
    //
    // See RFC 1210 for more details and justification.

    let impl1_substs = skolemizing_subst_for_impl(tcx, impl1_def_id);
    let (impl1_trait_ref, impl1_obligations) = {
        let selcx = &mut SelectionContext::new(&infcx);
        util::impl_trait_ref_and_oblig(selcx, impl1_def_id, &impl1_substs)
    };

    let impl1_predicates: Vec<_> = impl1_obligations.iter()
        .cloned()
        .map(|oblig| oblig.predicate)
        .collect();

    let penv = ty::ParameterEnvironment {
        tcx: tcx,
        free_substs: impl1_substs,
        implicit_region_bound: ty::ReEmpty, // FIXME: is this OK?
        caller_bounds: impl1_predicates,
        selection_cache: traits::SelectionCache::new(),
        evaluation_cache: traits::EvaluationCache::new(),
        free_id_outlive: region::DUMMY_CODE_EXTENT, // FIXME: is this OK?
    };

    // FIXME: unclear what `errors_will_be_reported` should be here...
    let infcx = infer::new_infer_ctxt(tcx, infcx.tables, Some(penv), true);
    let selcx = &mut SelectionContext::new(&infcx);

    let impl2_substs = util::fresh_type_vars_for_impl(&infcx, DUMMY_SP, impl2_def_id);
    let (impl2_trait_ref, impl2_obligations) =
        util::impl_trait_ref_and_oblig(selcx, impl2_def_id, &impl2_substs);

    // do the impls unify? If not, no specialization.
    if let Err(_) = infer::mk_eq_trait_refs(&infcx,
                                            true,
                                            TypeOrigin::Misc(DUMMY_SP),
                                            impl1_trait_ref,
                                            impl2_trait_ref) {
        debug!("specializes: {:?} does not unify with {:?}",
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
        debug!("specializes: for impls on {:?} and {:?}, could not fulfill: {:?} given {:?}",
               impl1_trait_ref,
               impl2_trait_ref,
               errors,
               infcx.parameter_environment.caller_bounds);
        return false;
    }

    debug!("specializes: an impl for {:?} specializes {:?} (`where` clauses elided)",
           impl1_trait_ref,
           impl2_trait_ref);
    true
}
