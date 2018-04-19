// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::traits::{EvaluationResult, Obligation, ObligationCause,
                    OverflowError, SelectionContext, TraitQueryMode};
use rustc::traits::query::CanonicalPredicateGoal;
use rustc::ty::{ParamEnvAnd, TyCtxt};
use syntax::codemap::DUMMY_SP;

crate fn evaluate_obligation<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    goal: CanonicalPredicateGoal<'tcx>,
) -> Result<EvaluationResult, OverflowError> {
    tcx.infer_ctxt().enter(|ref infcx| {
        let (
            ParamEnvAnd {
                param_env,
                value: predicate,
            },
            _canonical_inference_vars,
        ) = infcx.instantiate_canonical_with_fresh_inference_vars(DUMMY_SP, &goal);

        let mut selcx = SelectionContext::with_query_mode(&infcx, TraitQueryMode::Canonical);
        let obligation = Obligation::new(ObligationCause::dummy(), param_env, predicate);

        selcx.evaluate_obligation_recursively(&obligation)
    })
}
