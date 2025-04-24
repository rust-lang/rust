use rustc_hir::def_id::{CRATE_DEF_ID, DefId};
use rustc_infer::traits::Obligation;
use rustc_middle::traits::query::NoSolution;
pub use rustc_middle::traits::query::type_op::AscribeUserType;
use rustc_middle::traits::{ObligationCause, ObligationCauseCode};
use rustc_middle::ty::{self, ParamEnvAnd, Ty, TyCtxt, UserArgs, UserSelfTy, UserTypeKind};
use rustc_span::{DUMMY_SP, Span};
use tracing::{debug, instrument};

use crate::infer::canonical::{CanonicalQueryInput, CanonicalQueryResponse};
use crate::traits::ObligationCtxt;

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
        canonicalized: CanonicalQueryInput<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Result<CanonicalQueryResponse<'tcx, ()>, NoSolution> {
        tcx.type_op_ascribe_user_type(canonicalized)
    }

    fn perform_locally_with_next_solver(
        ocx: &ObligationCtxt<'_, 'tcx>,
        key: ParamEnvAnd<'tcx, Self>,
        span: Span,
    ) -> Result<Self::QueryResponse, NoSolution> {
        type_op_ascribe_user_type_with_span(ocx, key, span)
    }
}

/// The core of the `type_op_ascribe_user_type` query: for diagnostics purposes in NLL HRTB errors,
/// this query can be re-run to better track the span of the obligation cause, and improve the error
/// message. Do not call directly unless you're in that very specific context.
pub fn type_op_ascribe_user_type_with_span<'tcx>(
    ocx: &ObligationCtxt<'_, 'tcx>,
    key: ParamEnvAnd<'tcx, AscribeUserType<'tcx>>,
    span: Span,
) -> Result<(), NoSolution> {
    let (param_env, AscribeUserType { mir_ty, user_ty }) = key.into_parts();
    debug!("type_op_ascribe_user_type: mir_ty={:?} user_ty={:?}", mir_ty, user_ty);
    match user_ty.kind {
        UserTypeKind::Ty(user_ty) => relate_mir_and_user_ty(ocx, param_env, span, mir_ty, user_ty)?,
        UserTypeKind::TypeOf(def_id, user_args) => {
            relate_mir_and_user_args(ocx, param_env, span, mir_ty, def_id, user_args)?
        }
    };

    // Enforce any bounds that come from impl trait in bindings.
    ocx.register_obligations(user_ty.bounds.iter().map(|clause| {
        Obligation::new(ocx.infcx.tcx, ObligationCause::dummy_with_span(span), param_env, clause)
    }));

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
    ocx.register_obligation(Obligation::new(
        ocx.infcx.tcx,
        cause.clone(),
        param_env,
        ty::ClauseKind::WellFormed(user_ty.into()),
    ));

    let user_ty = ocx.normalize(&cause, param_env, user_ty);
    ocx.eq(&cause, param_env, mir_ty, user_ty)?;

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
        let instantiated_predicate = ocx.normalize(&cause, param_env, instantiated_predicate);

        ocx.register_obligation(Obligation::new(tcx, cause, param_env, instantiated_predicate));
    }

    // Now prove the well-formedness of `def_id` with `args`.
    // Note for some items, proving the WF of `ty` is not sufficient because the
    // well-formedness of an item may depend on the WF of gneneric args not present in the
    // item's type. Currently this is true for associated consts, e.g.:
    // ```rust
    // impl<T> MyTy<T> {
    //     const CONST: () = { /* arbitrary code that depends on T being WF */ };
    // }
    // ```
    for arg in args {
        ocx.register_obligation(Obligation::new(
            tcx,
            cause.clone(),
            param_env,
            ty::ClauseKind::WellFormed(arg),
        ));
    }

    if let Some(UserSelfTy { impl_def_id, self_ty }) = user_self_ty {
        ocx.register_obligation(Obligation::new(
            tcx,
            cause.clone(),
            param_env,
            ty::ClauseKind::WellFormed(self_ty.into()),
        ));

        let self_ty = ocx.normalize(&cause, param_env, self_ty);
        let impl_self_ty = tcx.type_of(impl_def_id).instantiate(tcx, args);
        let impl_self_ty = ocx.normalize(&cause, param_env, impl_self_ty);

        ocx.eq(&cause, param_env, self_ty, impl_self_ty)?;
    }

    Ok(())
}
