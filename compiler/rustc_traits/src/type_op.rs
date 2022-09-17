use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::at::ToTrace;
use rustc_infer::infer::canonical::{Canonical, QueryResponse};
use rustc_infer::infer::{DefiningAnchor, TyCtxtInferExt};
use rustc_infer::traits::ObligationCauseCode;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, FnSig, Lift, PolyFnSig, Ty, TyCtxt, TypeFoldable};
use rustc_middle::ty::{ParamEnvAnd, Predicate, ToPredicate};
use rustc_middle::ty::{UserSelfTy, UserSubsts};
use rustc_span::{Span, DUMMY_SP};
use rustc_trait_selection::infer::InferCtxtBuilderExt;
use rustc_trait_selection::traits::query::normalize::AtExt;
use rustc_trait_selection::traits::query::type_op::ascribe_user_type::AscribeUserType;
use rustc_trait_selection::traits::query::type_op::eq::Eq;
use rustc_trait_selection::traits::query::type_op::normalize::Normalize;
use rustc_trait_selection::traits::query::type_op::prove_predicate::ProvePredicate;
use rustc_trait_selection::traits::query::type_op::subtype::Subtype;
use rustc_trait_selection::traits::query::{Fallible, NoSolution};
use rustc_trait_selection::traits::{Normalized, Obligation, ObligationCause, ObligationCtxt};
use std::fmt;
use std::iter::zip;

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

/// The core of the `type_op_ascribe_user_type` query: for diagnostics purposes in NLL HRTB errors,
/// this query can be re-run to better track the span of the obligation cause, and improve the error
/// message. Do not call directly unless you're in that very specific context.
pub fn type_op_ascribe_user_type_with_span<'tcx>(
    ocx: &ObligationCtxt<'_, 'tcx>,
    key: ParamEnvAnd<'tcx, AscribeUserType<'tcx>>,
    span: Option<Span>,
) -> Result<(), NoSolution> {
    let (param_env, AscribeUserType { mir_ty, def_id, user_substs }) = key.into_parts();
    debug!(
        "type_op_ascribe_user_type: mir_ty={:?} def_id={:?} user_substs={:?}",
        mir_ty, def_id, user_substs
    );
    let cx = AscribeUserTypeCx { ocx, param_env, span: span.unwrap_or(DUMMY_SP) };
    cx.relate_mir_and_user_ty(mir_ty, def_id, user_substs)?;
    Ok(())
}

struct AscribeUserTypeCx<'me, 'tcx> {
    ocx: &'me ObligationCtxt<'me, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    span: Span,
}

impl<'me, 'tcx> AscribeUserTypeCx<'me, 'tcx> {
    fn normalize<T>(&self, value: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.normalize_with_cause(value, ObligationCause::misc(self.span, hir::CRATE_HIR_ID))
    }

    fn normalize_with_cause<T>(&self, value: T, cause: ObligationCause<'tcx>) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.ocx.normalize(cause, self.param_env, value)
    }

    fn eq<T>(&self, a: T, b: T) -> Result<(), NoSolution>
    where
        T: ToTrace<'tcx>,
    {
        Ok(self.ocx.eq(&ObligationCause::dummy_with_span(self.span), self.param_env, a, b)?)
    }

    fn prove_predicate(&self, predicate: Predicate<'tcx>, cause: ObligationCause<'tcx>) {
        self.ocx.register_obligation(Obligation::new(
            self.ocx.infcx.tcx,
            cause,
            self.param_env,
            predicate,
        ));
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.ocx.infcx.tcx
    }

    #[instrument(level = "debug", skip(self))]
    fn relate_mir_and_user_ty(
        &self,
        mir_ty: Ty<'tcx>,
        def_id: DefId,
        user_substs: UserSubsts<'tcx>,
    ) -> Result<(), NoSolution> {
        let UserSubsts { user_self_ty, substs } = user_substs;
        let tcx = self.tcx();

        let ty = tcx.bound_type_of(def_id).subst(tcx, substs);
        let ty = self.normalize(ty);
        debug!("relate_type_and_user_type: ty of def-id is {:?}", ty);

        self.eq(mir_ty, ty)?;

        // Prove the predicates coming along with `def_id`.
        //
        // Also, normalize the `instantiated_predicates`
        // because otherwise we wind up with duplicate "type
        // outlives" error messages.
        let instantiated_predicates = tcx.predicates_of(def_id).instantiate(tcx, substs);

        let cause = ObligationCause::dummy_with_span(self.span);

        debug!(?instantiated_predicates);
        for (instantiated_predicate, predicate_span) in
            zip(instantiated_predicates.predicates, instantiated_predicates.spans)
        {
            let span = if self.span == DUMMY_SP { predicate_span } else { self.span };
            let cause = ObligationCause::new(
                span,
                hir::CRATE_HIR_ID,
                ObligationCauseCode::AscribeUserTypeProvePredicate(predicate_span),
            );
            let instantiated_predicate =
                self.normalize_with_cause(instantiated_predicate, cause.clone());
            self.prove_predicate(instantiated_predicate, cause);
        }

        if let Some(UserSelfTy { impl_def_id, self_ty }) = user_self_ty {
            let self_ty = self.normalize(self_ty);
            let impl_self_ty = tcx.bound_type_of(impl_def_id).subst(tcx, substs);
            let impl_self_ty = self.normalize(impl_self_ty);

            self.eq(self_ty, impl_self_ty)?;

            self.prove_predicate(
                ty::Binder::dummy(ty::PredicateKind::WellFormed(impl_self_ty.into()))
                    .to_predicate(tcx),
                cause.clone(),
            );
        }

        // In addition to proving the predicates, we have to
        // prove that `ty` is well-formed -- this is because
        // the WF of `ty` is predicated on the substs being
        // well-formed, and we haven't proven *that*. We don't
        // want to prove the WF of types from  `substs` directly because they
        // haven't been normalized.
        //
        // FIXME(nmatsakis): Well, perhaps we should normalize
        // them?  This would only be relevant if some input
        // type were ill-formed but did not appear in `ty`,
        // which...could happen with normalization...
        self.prove_predicate(
            ty::Binder::dummy(ty::PredicateKind::WellFormed(ty.into())).to_predicate(tcx),
            cause,
        );
        Ok(())
    }
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
) -> Fallible<T>
where
    T: fmt::Debug + TypeFoldable<'tcx> + Lift<'tcx>,
{
    let (param_env, Normalize { value }) = key.into_parts();
    let Normalized { value, obligations } =
        ocx.infcx.at(&ObligationCause::dummy(), param_env).normalize(value)?;
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
