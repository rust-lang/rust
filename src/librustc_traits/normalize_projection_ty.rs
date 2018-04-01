// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::infer::canonical::{Canonical, QueryResult};
use rustc::traits::{self, FulfillmentContext, Normalized, ObligationCause,
                    SelectionContext};
use rustc::traits::query::{CanonicalProjectionGoal, NoSolution, normalize::NormalizationResult};
use rustc::ty::{ParamEnvAnd, TyCtxt};
use rustc_data_structures::sync::Lrc;
use syntax::ast::DUMMY_NODE_ID;
use syntax_pos::DUMMY_SP;
use util;
use std::sync::atomic::Ordering;

crate fn normalize_projection_ty<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    goal: CanonicalProjectionGoal<'tcx>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, NormalizationResult<'tcx>>>>, NoSolution> {
    debug!("normalize_provider(goal={:#?})", goal);

    tcx.sess.perf_stats.normalize_projection_ty.fetch_add(1, Ordering::Relaxed);
    tcx.infer_ctxt().enter(|ref infcx| {
        let (
            ParamEnvAnd {
                param_env,
                value: goal,
            },
            canonical_inference_vars,
        ) = infcx.instantiate_canonical_with_fresh_inference_vars(DUMMY_SP, &goal);
        let fulfill_cx = &mut FulfillmentContext::new();
        let selcx = &mut SelectionContext::new(infcx);
        let cause = ObligationCause::misc(DUMMY_SP, DUMMY_NODE_ID);
        let Normalized {
            value: answer,
            obligations,
        } = traits::normalize_projection_type(selcx, param_env, goal, cause, 0);
        fulfill_cx.register_predicate_obligations(infcx, obligations);

        // Now that we have fulfilled as much as we can, create a solution
        // from what we've learned.
        util::make_query_response(
            infcx,
            canonical_inference_vars,
            NormalizationResult { normalized_ty: answer },
            fulfill_cx,
        )
    })
}
