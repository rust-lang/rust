use rustc::ty::query::Providers;
use rustc::ty::subst::{GenericArg, Subst, UserSelfTy, UserSubsts};
use rustc::ty::{
    FnSig, Lift, ParamEnv, ParamEnvAnd, PolyFnSig, Predicate, Ty, TyCtxt, TypeFoldable, Variance,
};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::at::ToTrace;
use rustc_infer::infer::canonical::{Canonical, QueryResponse};
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_infer::traits::TraitEngineExt as _;
use rustc_span::DUMMY_SP;
use rustc_trait_selection::infer::InferCtxtBuilderExt;
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::query::normalize::AtExt;
use rustc_trait_selection::traits::query::type_op::ascribe_user_type::AscribeUserType;
use rustc_trait_selection::traits::query::type_op::eq::Eq;
use rustc_trait_selection::traits::query::type_op::normalize::Normalize;
use rustc_trait_selection::traits::query::type_op::prove_predicate::ProvePredicate;
use rustc_trait_selection::traits::query::type_op::subtype::Subtype;
use rustc_trait_selection::traits::query::{Fallible, NoSolution};
use rustc_trait_selection::traits::{Normalized, Obligation, ObligationCause, TraitEngine};
use std::fmt;

crate fn provide(p: &mut Providers<'_>) {
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
    tcx.infer_ctxt().enter_canonical_trait_query(&canonicalized, |infcx, fulfill_cx, key| {
        let (param_env, AscribeUserType { mir_ty, def_id, user_substs }) = key.into_parts();

        debug!(
            "type_op_ascribe_user_type: mir_ty={:?} def_id={:?} user_substs={:?}",
            mir_ty, def_id, user_substs
        );

        let mut cx = AscribeUserTypeCx { infcx, param_env, fulfill_cx };
        cx.relate_mir_and_user_ty(mir_ty, def_id, user_substs)?;

        Ok(())
    })
}

struct AscribeUserTypeCx<'me, 'tcx> {
    infcx: &'me InferCtxt<'me, 'tcx>,
    param_env: ParamEnv<'tcx>,
    fulfill_cx: &'me mut dyn TraitEngine<'tcx>,
}

impl AscribeUserTypeCx<'me, 'tcx> {
    fn normalize<T>(&mut self, value: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.infcx
            .partially_normalize_associated_types_in(
                DUMMY_SP,
                hir::CRATE_HIR_ID,
                self.param_env,
                &value,
            )
            .into_value_registering_obligations(self.infcx, self.fulfill_cx)
    }

    fn relate<T>(&mut self, a: T, variance: Variance, b: T) -> Result<(), NoSolution>
    where
        T: ToTrace<'tcx>,
    {
        Ok(self
            .infcx
            .at(&ObligationCause::dummy(), self.param_env)
            .relate(a, variance, b)?
            .into_value_registering_obligations(self.infcx, self.fulfill_cx))
    }

    fn prove_predicate(&mut self, predicate: Predicate<'tcx>) {
        self.fulfill_cx.register_predicate_obligation(
            self.infcx,
            Obligation::new(ObligationCause::dummy(), self.param_env, predicate),
        );
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn subst<T>(&self, value: T, substs: &[GenericArg<'tcx>]) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        value.subst(self.tcx(), substs)
    }

    fn relate_mir_and_user_ty(
        &mut self,
        mir_ty: Ty<'tcx>,
        def_id: DefId,
        user_substs: UserSubsts<'tcx>,
    ) -> Result<(), NoSolution> {
        let UserSubsts { user_self_ty, substs } = user_substs;
        let tcx = self.tcx();

        let ty = tcx.type_of(def_id);
        let ty = self.subst(ty, substs);
        debug!("relate_type_and_user_type: ty of def-id is {:?}", ty);
        let ty = self.normalize(ty);

        self.relate(mir_ty, Variance::Invariant, ty)?;

        // Prove the predicates coming along with `def_id`.
        //
        // Also, normalize the `instantiated_predicates`
        // because otherwise we wind up with duplicate "type
        // outlives" error messages.
        let instantiated_predicates =
            self.tcx().predicates_of(def_id).instantiate(self.tcx(), substs);
        for instantiated_predicate in instantiated_predicates.predicates {
            let instantiated_predicate = self.normalize(instantiated_predicate);
            self.prove_predicate(instantiated_predicate);
        }

        if let Some(UserSelfTy { impl_def_id, self_ty }) = user_self_ty {
            let impl_self_ty = self.tcx().type_of(impl_def_id);
            let impl_self_ty = self.subst(impl_self_ty, &substs);
            let impl_self_ty = self.normalize(impl_self_ty);

            self.relate(self_ty, Variance::Invariant, impl_self_ty)?;

            self.prove_predicate(Predicate::WellFormed(impl_self_ty));
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
        self.prove_predicate(Predicate::WellFormed(ty));
        Ok(())
    }
}

fn type_op_eq<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Eq<'tcx>>>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, ()>>, NoSolution> {
    tcx.infer_ctxt().enter_canonical_trait_query(&canonicalized, |infcx, fulfill_cx, key| {
        let (param_env, Eq { a, b }) = key.into_parts();
        Ok(infcx
            .at(&ObligationCause::dummy(), param_env)
            .eq(a, b)?
            .into_value_registering_obligations(infcx, fulfill_cx))
    })
}

fn type_op_normalize<T>(
    infcx: &InferCtxt<'_, 'tcx>,
    fulfill_cx: &mut dyn TraitEngine<'tcx>,
    key: ParamEnvAnd<'tcx, Normalize<T>>,
) -> Fallible<T>
where
    T: fmt::Debug + TypeFoldable<'tcx> + Lift<'tcx>,
{
    let (param_env, Normalize { value }) = key.into_parts();
    let Normalized { value, obligations } =
        infcx.at(&ObligationCause::dummy(), param_env).normalize(&value)?;
    fulfill_cx.register_predicate_obligations(infcx, obligations);
    Ok(value)
}

fn type_op_normalize_ty(
    tcx: TyCtxt<'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<Ty<'tcx>>>>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, Ty<'tcx>>>, NoSolution> {
    tcx.infer_ctxt().enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_normalize_predicate(
    tcx: TyCtxt<'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<Predicate<'tcx>>>>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, Predicate<'tcx>>>, NoSolution> {
    tcx.infer_ctxt().enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_normalize_fn_sig(
    tcx: TyCtxt<'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<FnSig<'tcx>>>>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, FnSig<'tcx>>>, NoSolution> {
    tcx.infer_ctxt().enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_normalize_poly_fn_sig(
    tcx: TyCtxt<'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<PolyFnSig<'tcx>>>>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, PolyFnSig<'tcx>>>, NoSolution> {
    tcx.infer_ctxt().enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_subtype<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Subtype<'tcx>>>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, ()>>, NoSolution> {
    tcx.infer_ctxt().enter_canonical_trait_query(&canonicalized, |infcx, fulfill_cx, key| {
        let (param_env, Subtype { sub, sup }) = key.into_parts();
        Ok(infcx
            .at(&ObligationCause::dummy(), param_env)
            .sup(sup, sub)?
            .into_value_registering_obligations(infcx, fulfill_cx))
    })
}

fn type_op_prove_predicate<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, ProvePredicate<'tcx>>>,
) -> Result<&'tcx Canonical<'tcx, QueryResponse<'tcx, ()>>, NoSolution> {
    tcx.infer_ctxt().enter_canonical_trait_query(&canonicalized, |infcx, fulfill_cx, key| {
        let (param_env, ProvePredicate { predicate }) = key.into_parts();
        fulfill_cx.register_predicate_obligation(
            infcx,
            Obligation::new(ObligationCause::dummy(), param_env, predicate),
        );
        Ok(())
    })
}
