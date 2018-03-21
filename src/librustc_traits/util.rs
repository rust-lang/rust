// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::infer::InferCtxt;
use rustc::infer::canonical::{CanonicalVarValues, Canonicalize, Certainty, QueryRegionConstraints,
                              QueryResult};
use rustc::infer::region_constraints::{Constraint, RegionConstraintData};
use rustc::traits::FulfillmentContext;
use rustc::traits::query::NoSolution;
use rustc::ty;
use std::fmt::Debug;

/// The canonicalization form of `QueryResult<'tcx, T>`.
type CanonicalizedQueryResult<'gcx, 'tcx, T> =
    <QueryResult<'tcx, T> as Canonicalize<'gcx, 'tcx>>::Canonicalized;

crate fn make_query_response<'gcx, 'tcx, T>(
    infcx: &InferCtxt<'_, 'gcx, 'tcx>,
    inference_vars: CanonicalVarValues<'tcx>,
    answer: T,
    fulfill_cx: &mut FulfillmentContext<'tcx>,
) -> Result<CanonicalizedQueryResult<'gcx, 'tcx, T>, NoSolution>
where
    T: Debug,
    QueryResult<'tcx, T>: Canonicalize<'gcx, 'tcx>,
{
    let tcx = infcx.tcx;

    debug!(
        "make_query_response(\
         inference_vars={:?}, \
         answer={:?})",
        inference_vars, answer,
    );

    // Select everything, returning errors.
    let true_errors = match fulfill_cx.select_where_possible(infcx) {
        Ok(()) => vec![],
        Err(errors) => errors,
    };
    debug!("true_errors = {:#?}", true_errors);

    if !true_errors.is_empty() {
        // FIXME -- we don't indicate *why* we failed to solve
        debug!("make_query_response: true_errors={:#?}", true_errors);
        return Err(NoSolution);
    }

    // Anything left unselected *now* must be an ambiguity.
    let ambig_errors = match fulfill_cx.select_all_or_error(infcx) {
        Ok(()) => vec![],
        Err(errors) => errors,
    };
    debug!("ambig_errors = {:#?}", ambig_errors);

    let region_obligations = infcx.take_registered_region_obligations();

    let (region_outlives, ty_outlives) = infcx.with_region_constraints(|region_constraints| {
        let RegionConstraintData {
            constraints,
            verifys,
            givens,
        } = region_constraints;

        assert!(verifys.is_empty());
        assert!(givens.is_empty());

        let region_outlives: Vec<_> = constraints
            .into_iter()
            .map(|(k, _)| match *k {
                Constraint::VarSubVar(v1, v2) => {
                    (tcx.mk_region(ty::ReVar(v1)), tcx.mk_region(ty::ReVar(v2)))
                }
                Constraint::VarSubReg(v1, r2) => (tcx.mk_region(ty::ReVar(v1)), r2),
                Constraint::RegSubVar(r1, v2) => (r1, tcx.mk_region(ty::ReVar(v2))),
                Constraint::RegSubReg(r1, r2) => (r1, r2),
            })
            .collect();

        let ty_outlives: Vec<_> = region_obligations
            .into_iter()
            .map(|(_, r_o)| (r_o.sup_type, r_o.sub_region))
            .collect();

        (region_outlives, ty_outlives)
    });

    let certainty = if ambig_errors.is_empty() {
        Certainty::Proven
    } else {
        Certainty::Ambiguous
    };

    let (canonical_result, _) = infcx.canonicalize_response(&QueryResult {
        var_values: inference_vars,
        region_constraints: QueryRegionConstraints {
            region_outlives,
            ty_outlives,
        },
        certainty,
        value: answer,
    });

    debug!(
        "make_query_response: canonical_result = {:#?}",
        canonical_result
    );

    Ok(canonical_result)
}
