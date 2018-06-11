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
use rustc::infer::InferCtxt;
use rustc::traits::query::type_op::normalize::Normalize;
use rustc::traits::query::NoSolution;
use rustc::traits::{FulfillmentContext, Normalized, ObligationCause};
use rustc::ty::{FnSig, Lift, PolyFnSig, Predicate, Ty, TyCtxt, TypeFoldable};
use rustc_data_structures::sync::Lrc;
use std::fmt;
use syntax::codemap::DUMMY_SP;

fn type_op_normalize<'gcx, 'tcx, T>(
    infcx: &InferCtxt<'_, 'gcx, 'tcx>,
    canonicalized: Canonical<'tcx, Normalize<'tcx, T>>,
) -> Result<Lrc<Canonical<'gcx, QueryResult<'gcx, <T as Lift<'gcx>>::Lifted>>>, NoSolution>
where
    T: fmt::Debug + TypeFoldable<'tcx> + Lift<'gcx>,
{
    let (Normalize { param_env, value }, canonical_inference_vars) =
        infcx.instantiate_canonical_with_fresh_inference_vars(DUMMY_SP, &canonicalized);
    let fulfill_cx = &mut FulfillmentContext::new();
    let Normalized { value, obligations } = infcx
        .at(&ObligationCause::dummy(), param_env)
        .normalize(&value)?;
    fulfill_cx.register_predicate_obligations(infcx, obligations);
    infcx.make_canonicalized_query_result(canonical_inference_vars, value, fulfill_cx)
}

crate fn type_op_normalize_ty<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, Normalize<'tcx, Ty<'tcx>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, Ty<'tcx>>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter(|ref infcx| type_op_normalize(infcx, canonicalized))
}

crate fn type_op_normalize_predicate<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, Normalize<'tcx, Predicate<'tcx>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, Predicate<'tcx>>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter(|ref infcx| type_op_normalize(infcx, canonicalized))
}

crate fn type_op_normalize_fn_sig<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, Normalize<'tcx, FnSig<'tcx>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, FnSig<'tcx>>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter(|ref infcx| type_op_normalize(infcx, canonicalized))
}

crate fn type_op_normalize_poly_fn_sig<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, Normalize<'tcx, PolyFnSig<'tcx>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResult<'tcx, PolyFnSig<'tcx>>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter(|ref infcx| type_op_normalize(infcx, canonicalized))
}
