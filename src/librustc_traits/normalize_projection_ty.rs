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
use rustc::infer::InferOk;
use rustc::traits::query::{normalize::NormalizationResult, CanonicalProjectionGoal, NoSolution};
use rustc::traits::{self, ObligationCause, SelectionContext};
use rustc::ty::{ParamEnvAnd, TyCtxt};
use rustc_data_structures::sync::Lrc;
use std::sync::atomic::Ordering;
use syntax::ast::DUMMY_NODE_ID;
use syntax_pos::DUMMY_SP;

crate fn normalize_projection_ty<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    goal: CanonicalProjectionGoal<'tcx>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, NormalizationResult<'tcx>>>>, NoSolution> {
    debug!("normalize_provider(goal={:#?})", goal);

    tcx.sess
        .perf_stats
        .normalize_projection_ty
        .fetch_add(1, Ordering::Relaxed);
    tcx.infer_ctxt().enter_canonical_trait_query(
        &goal,
        |infcx,
         ParamEnvAnd {
             param_env,
             value: goal,
         }| {
            let selcx = &mut SelectionContext::new(infcx);
            let cause = ObligationCause::misc(DUMMY_SP, DUMMY_NODE_ID);
            let mut obligations = vec![];
            let answer = traits::normalize_projection_type(
                selcx,
                param_env,
                goal,
                cause,
                0,
                &mut obligations,
            );
            Ok(InferOk {
                value: NormalizationResult {
                    normalized_ty: answer,
                },
                obligations,
            })
        },
    )
}
