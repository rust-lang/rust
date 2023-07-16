use crate::infer::canonical::{Canonical, CanonicalQueryResponse};
use crate::traits::ObligationCtxt;
use rustc_hir::def_id::{DefId, CRATE_DEF_ID};
use rustc_infer::traits::Obligation;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::{ObligationCause, ObligationCauseCode};
use rustc_middle::ty::{self, ParamEnvAnd, Ty, TyCtxt, UserArgs, UserSelfTy, UserType};

pub use rustc_middle::traits::query::type_op::AscribeUserType;
use rustc_span::{Span, DUMMY_SP};

impl<'tcx> super::QueryTypeOp<'tcx> for AscribeUserType<'tcx> {
    type QueryResponse = ();

    fn try_fast_path(
        _tcx: TyCtxt<'tcx>,
        _key: &ParamEnvAnd<'tcx, Self>,
    ) -> Option<Self::QueryResponse> {
        None
    }

    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Result<CanonicalQueryResponse<'tcx, ()>, NoSolution> {
        tcx.type_op_ascribe_user_type(canonicalized)
    }

    fn perform_locally_in_new_solver(
        ocx: &ObligationCtxt<'_, 'tcx>,
        key: ParamEnvAnd<'tcx, Self>,
    ) -> Result<Self::QueryResponse, NoSolution> {
        type_op_ascribe_user_type_with_span(ocx, key, None)
    }
}

/// The core of the `type_op_ascribe_user_type` query: for diagnostics purposes in NLL HRTB errors,
/// this query can be re-run to better track the span of the obligation cause, and improve the error
/// message. Do not call directly unless you're in that very specific context.
pub fn type_op_ascribe_user_type_with_span<'tcx>(
    ocx: &ObligationCtxt<'_, 'tcx>,
    key: ParamEnvAnd<'tcx, AscribeUserType<'tcx>>,
    span: Option<Span>,
) -> Result<(), NoSolution> {
    let (param_env, AscribeUserType { mir_ty, user_ty }) = key.into_parts();
    debug!("type_op_ascribe_user_type: mir_ty={:?} user_ty={:?}", mir_ty, user_ty);
    let span = span.unwrap_or(DUMMY_SP);
    match user_ty {
        UserType::Ty(user_ty) => relate_mir_and_user_ty(ocx, param_env, span, mir_ty, user_ty)?,
        UserType::TypeOf(def_id, user_args) => {
            relate_mir_and_user_args(ocx, param_env, span, mir_ty, def_id, user_args)?
        }
    };
    Ok(())
}

#[instrument(level = "debug", skip(ocx, param_env, span))]
fn relate_mir_and_user_ty<'tcx>(
    ocx: &ObligationCtxt<'_, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    span: Span,
    mir_ty: Ty<'tcx>,
    user_ty: Ty<'tcx>,
) -> Result<(), NoSolution> {
    let cause = ObligationCause::dummy_with_span(span);
    let user_ty = ocx.normalize(&cause, param_env, user_ty);
    ocx.eq(&cause, param_env, mir_ty, user_ty)?;

    // FIXME(#104764): We should check well-formedness before normalization.
    let predicate =
        ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(user_ty.into())));
    ocx.register_obligation(Obligation::new(ocx.infcx.tcx, cause, param_env, predicate));
    Ok(())
}

#[instrument(level = "debug", skip(ocx, param_env, span))]
fn relate_mir_and_user_args<'tcx>(
    ocx: &ObligationCtxt<'_, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    span: Span,
    mir_ty: Ty<'tcx>,
    def_id: DefId,
    user_args: UserArgs<'tcx>,
) -> Result<(), NoSolution> {
    let param_env = param_env.without_const();
    let UserArgs { user_self_ty, args } = user_args;
    let tcx = ocx.infcx.tcx;
    let cause = ObligationCause::dummy_with_span(span);

    let ty = tcx.type_of(def_id).instantiate(tcx, args);
    let ty = ocx.normalize(&cause, param_env, ty);
    debug!("relate_type_and_user_type: ty of def-id is {:?}", ty);

    ocx.eq(&cause, param_env, mir_ty, ty)?;

    // Prove the predicates coming along with `def_id`.
    //
    // Also, normalize the `instantiated_predicates`
    // because otherwise we wind up with duplicate "type
    // outlives" error messages.
    let instantiated_predicates = tcx.predicates_of(def_id).instantiate(tcx, args);

    debug!(?instantiated_predicates);
    for (instantiated_predicate, predicate_span) in instantiated_predicates {
        let span = if span == DUMMY_SP { predicate_span } else { span };
        let cause = ObligationCause::new(
            span,
            CRATE_DEF_ID,
            ObligationCauseCode::AscribeUserTypeProvePredicate(predicate_span),
        );
        let instantiated_predicate =
            ocx.normalize(&cause.clone(), param_env, instantiated_predicate);

        ocx.register_obligation(Obligation::new(tcx, cause, param_env, instantiated_predicate));
    }

    if let Some(UserSelfTy { impl_def_id, self_ty }) = user_self_ty {
        let self_ty = ocx.normalize(&cause, param_env, self_ty);
        let impl_self_ty = tcx.type_of(impl_def_id).instantiate(tcx, args);
        let impl_self_ty = ocx.normalize(&cause, param_env, impl_self_ty);

        ocx.eq(&cause, param_env, self_ty, impl_self_ty)?;
        let predicate = ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(
            impl_self_ty.into(),
        )));
        ocx.register_obligation(Obligation::new(tcx, cause.clone(), param_env, predicate));
    }

    // In addition to proving the predicates, we have to
    // prove that `ty` is well-formed -- this is because
    // the WF of `ty` is predicated on the args being
    // well-formed, and we haven't proven *that*. We don't
    // want to prove the WF of types from  `args` directly because they
    // haven't been normalized.
    //
    // FIXME(nmatsakis): Well, perhaps we should normalize
    // them?  This would only be relevant if some input
    // type were ill-formed but did not appear in `ty`,
    // which...could happen with normalization...
    let predicate =
        ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(ty.into())));
    ocx.register_obligation(Obligation::new(tcx, cause, param_env, predicate));
    Ok(())
}
