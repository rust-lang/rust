use rustc_infer::infer::canonical::{Canonical, QueryResponse};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::query::Providers;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::DefiningAnchor;
use rustc_middle::ty::{FnSig, Lift, PolyFnSig, Ty, TyCtxt, TypeFoldable};
use rustc_middle::ty::{ParamEnvAnd, Predicate};
use rustc_trait_selection::infer::InferCtxtBuilderExt;
use rustc_trait_selection::traits::query::normalize::QueryNormalizeExt;
use rustc_trait_selection::traits::query::type_op::ascribe_user_type::{
    type_op_ascribe_user_type_with_span, AscribeUserType,
};
use rustc_trait_selection::traits::query::type_op::eq::Eq;
use rustc_trait_selection::traits::query::type_op::normalize::Normalize;
use rustc_trait_selection::traits::query::type_op::prove_predicate::ProvePredicate;
use rustc_trait_selection::traits::query::type_op::subtype::Subtype;
use rustc_trait_selection::traits::{Normalized, Obligation, ObligationCause, ObligationCtxt};
use std::fmt;

pub(crate) fn provide(p: &mut Providers) {
    *p = Providers {
        type_op_ascribe_user_type,
        type_op_eq,
        type_op_prove_predicate,
        type_op_subtype,
        type_op_normalize_ty,
        type_op_normalize_predicate,
        type_op_normalize_fn_sig,
        type_op_normalize_poly_fn_sig,
        ..*p
    };
}

fn type_op_ascribe_user_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, AscribeUserType<'tcx>>>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, ()>>, NoSolution> {
    tcx.infer_ctxt().enter_canonical_trait_query(&canonicalized, |ocx, key| {
        type_op_ascribe_user_type_with_span(ocx, key, None)
    })
}

fn type_op_eq<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Eq<'tcx>>>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, ()>>, NoSolution> {
    tcx.infer_ctxt().enter_canonical_trait_query(&canonicalized, |ocx, key| {
        let (param_env, Eq { a, b }) = key.into_parts();
        Ok(ocx.eq(&ObligationCause::dummy(), param_env, a, b)?)
    })
}

fn type_op_normalize<'tcx, T>(
    ocx: &ObligationCtxt<'_, 'tcx>,
    key: ParamEnvAnd<'tcx, Normalize<T>>,
) -> Result<T, NoSolution>
where
    T: fmt::Debug + TypeFoldable<TyCtxt<'tcx>> + Lift<'tcx>,
{
    let (param_env, Normalize { value }) = key.into_parts();
    let Normalized { value, obligations } =
        ocx.infcx.at(&ObligationCause::dummy(), param_env).query_normalize(value)?;
    ocx.register_obligations(obligations);
    Ok(value)
}

fn type_op_normalize_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<Ty<'tcx>>>>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, Ty<'tcx>>>, NoSolution> {
    tcx.infer_ctxt().enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_normalize_predicate<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<Predicate<'tcx>>>>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, Predicate<'tcx>>>, NoSolution> {
    tcx.infer_ctxt().enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_normalize_fn_sig<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<FnSig<'tcx>>>>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, FnSig<'tcx>>>, NoSolution> {
    tcx.infer_ctxt().enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_normalize_poly_fn_sig<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<PolyFnSig<'tcx>>>>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, PolyFnSig<'tcx>>>, NoSolution> {
    tcx.infer_ctxt().enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_subtype<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Subtype<'tcx>>>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, ()>>, NoSolution> {
    tcx.infer_ctxt().enter_canonical_trait_query(&canonicalized, |ocx, key| {
        let (param_env, Subtype { sub, sup }) = key.into_parts();
        Ok(ocx.sup(&ObligationCause::dummy(), param_env, sup, sub)?)
    })
}

fn type_op_prove_predicate<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, ProvePredicate<'tcx>>>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, ()>>, NoSolution> {
    // HACK This bubble is required for this test to pass:
    // impl-trait/issue-99642.rs
    tcx.infer_ctxt().with_opaque_type_inference(DefiningAnchor::Bubble).enter_canonical_trait_query(
        &canonicalized,
        |ocx, key| {
            type_op_prove_predicate_with_cause(ocx, key, ObligationCause::dummy());
            Ok(())
        },
    )
}

/// The core of the `type_op_prove_predicate` query: for diagnostics purposes in NLL HRTB errors,
/// this query can be re-run to better track the span of the obligation cause, and improve the error
/// message. Do not call directly unless you're in that very specific context.
pub fn type_op_prove_predicate_with_cause<'tcx>(
    ocx: &ObligationCtxt<'_, 'tcx>,
    key: ParamEnvAnd<'tcx, ProvePredicate<'tcx>>,
    cause: ObligationCause<'tcx>,
) {
    let (param_env, ProvePredicate { predicate }) = key.into_parts();
    ocx.register_obligation(Obligation::new(ocx.infcx.tcx, cause, param_env, predicate));
}
