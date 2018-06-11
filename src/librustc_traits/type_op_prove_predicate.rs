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
use rustc::traits::query::type_op::prove_predicate::ProvePredicate;
use rustc::traits::query::NoSolution;
use rustc::traits::{FulfillmentContext, Obligation, ObligationCause, TraitEngine};
use rustc::ty::TyCtxt;
use rustc_data_structures::sync::Lrc;
use syntax::codemap::DUMMY_SP;

crate fn type_op_prove_predicate<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, ProvePredicate<'tcx>>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, ()>>>, NoSolution> {
    let tcx = tcx.global_tcx();
    tcx.infer_ctxt().enter(|ref infcx| {
        let (ProvePredicate { param_env, predicate }, canonical_inference_vars) =
            infcx.instantiate_canonical_with_fresh_inference_vars(DUMMY_SP, &canonicalized);
        let fulfill_cx = &mut FulfillmentContext::new();
        let obligation = Obligation::new(ObligationCause::dummy(), param_env, predicate);
        fulfill_cx.register_predicate_obligation(infcx, obligation);
        infcx.make_canonicalized_query_result(canonical_inference_vars, (), fulfill_cx)
    })
}
