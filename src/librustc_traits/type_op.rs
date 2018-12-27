use rustc::infer::at::ToTrace;
use rustc::infer::canonical::{Canonical, QueryResponse};
use rustc::infer::InferCtxt;
use rustc::hir::def_id::DefId;
use rustc::mir::ProjectionKind;
use rustc::mir::tcx::PlaceTy;
use rustc::traits::query::type_op::ascribe_user_type::AscribeUserType;
use rustc::traits::query::type_op::eq::Eq;
use rustc::traits::query::type_op::normalize::Normalize;
use rustc::traits::query::type_op::prove_predicate::ProvePredicate;
use rustc::traits::query::type_op::subtype::Subtype;
use rustc::traits::query::{Fallible, NoSolution};
use rustc::traits::{
    FulfillmentContext, Normalized, Obligation, ObligationCause, TraitEngine, TraitEngineExt,
};
use rustc::ty::query::Providers;
use rustc::ty::subst::{Kind, Subst, UserSelfTy, UserSubsts};
use rustc::ty::{
    FnSig, Lift, ParamEnv, ParamEnvAnd, PolyFnSig, Predicate, Ty, TyCtxt, TypeFoldable, Variance,
};
use rustc_data_structures::sync::Lrc;
use std::fmt;
use syntax::ast;
use syntax_pos::DUMMY_SP;

crate fn provide(p: &mut Providers) {
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
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, AscribeUserType<'tcx>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResponse<'tcx, ()>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter_canonical_trait_query(&canonicalized, |infcx, fulfill_cx, key| {
            let (
                param_env,
                AscribeUserType {
                    mir_ty,
                    variance,
                    def_id,
                    user_substs,
                    projs,
                },
            ) = key.into_parts();

            debug!(
                "type_op_ascribe_user_type(\
                 mir_ty={:?}, variance={:?}, def_id={:?}, user_substs={:?}, projs={:?}\
                 )",
                mir_ty, variance, def_id, user_substs, projs,
            );

            let mut cx = AscribeUserTypeCx {
                infcx,
                param_env,
                fulfill_cx,
            };
            cx.relate_mir_and_user_ty(mir_ty, variance, def_id, user_substs, projs)?;

            Ok(())
        })
}

struct AscribeUserTypeCx<'me, 'gcx: 'tcx, 'tcx: 'me> {
    infcx: &'me InferCtxt<'me, 'gcx, 'tcx>,
    param_env: ParamEnv<'tcx>,
    fulfill_cx: &'me mut FulfillmentContext<'tcx>,
}

impl AscribeUserTypeCx<'me, 'gcx, 'tcx> {
    fn normalize<T>(&mut self, value: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.infcx
            .partially_normalize_associated_types_in(
                DUMMY_SP,
                ast::CRATE_NODE_ID,
                self.param_env,
                &value,
            )
            .into_value_registering_obligations(self.infcx, self.fulfill_cx)
    }

    fn relate<T>(&mut self, a: T, variance: Variance, b: T) -> Result<(), NoSolution>
    where
        T: ToTrace<'tcx>,
    {
        Ok(self.infcx
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

    fn tcx(&self) -> TyCtxt<'me, 'gcx, 'tcx> {
        self.infcx.tcx
    }

    fn subst<T>(&self, value: T, substs: &[Kind<'tcx>]) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        value.subst(self.tcx(), substs)
    }

    fn relate_mir_and_user_ty(
        &mut self,
        mir_ty: Ty<'tcx>,
        variance: Variance,
        def_id: DefId,
        user_substs: UserSubsts<'tcx>,
        projs: &[ProjectionKind<'tcx>],
    ) -> Result<(), NoSolution> {
        let UserSubsts {
            substs,
            user_self_ty,
        } = user_substs;

        let tcx = self.tcx();

        let ty = tcx.type_of(def_id);
        let ty = self.subst(ty, substs);
        debug!("relate_type_and_user_type: ty of def-id is {:?}", ty);
        let ty = self.normalize(ty);

        // We need to follow any provided projetions into the type.
        //
        // if we hit a ty var as we descend, then just skip the
        // attempt to relate the mir local with any type.

        struct HitTyVar;
        let mut curr_projected_ty: Result<PlaceTy, HitTyVar>;
        curr_projected_ty = Ok(PlaceTy::from_ty(ty));
        for proj in projs {
            let projected_ty = if let Ok(projected_ty) = curr_projected_ty {
                projected_ty
            } else {
                break;
            };
            curr_projected_ty = projected_ty.projection_ty_core(
                tcx, proj, |this, field, &()| {
                    if this.to_ty(tcx).is_ty_var() {
                        Err(HitTyVar)
                    } else {
                        let ty = this.field_ty(tcx, field);
                        Ok(self.normalize(ty))
                    }
                });
        }

        if let Ok(projected_ty) = curr_projected_ty {
            let ty = projected_ty.to_ty(tcx);
            self.relate(mir_ty, variance, ty)?;
        }

        if let Some(UserSelfTy {
            impl_def_id,
            self_ty,
        }) = user_self_ty
        {
            let impl_self_ty = self.tcx().type_of(impl_def_id);
            let impl_self_ty = self.subst(impl_self_ty, &substs);
            let impl_self_ty = self.normalize(impl_self_ty);

            self.relate(self_ty, Variance::Invariant, impl_self_ty)?;

            self.prove_predicate(Predicate::WellFormed(impl_self_ty));
        }

        // Prove the predicates coming along with `def_id`.
        //
        // Also, normalize the `instantiated_predicates`
        // because otherwise we wind up with duplicate "type
        // outlives" error messages.
        let instantiated_predicates = self.tcx()
            .predicates_of(def_id)
            .instantiate(self.tcx(), substs);
        for instantiated_predicate in instantiated_predicates.predicates {
            let instantiated_predicate = self.normalize(instantiated_predicate);
            self.prove_predicate(instantiated_predicate);
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
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Eq<'tcx>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResponse<'tcx, ()>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter_canonical_trait_query(&canonicalized, |infcx, fulfill_cx, key| {
            let (param_env, Eq { a, b }) = key.into_parts();
            Ok(infcx
                .at(&ObligationCause::dummy(), param_env)
                .eq(a, b)?
                .into_value_registering_obligations(infcx, fulfill_cx))
        })
}

fn type_op_normalize<T>(
    infcx: &InferCtxt<'_, 'gcx, 'tcx>,
    fulfill_cx: &mut FulfillmentContext<'tcx>,
    key: ParamEnvAnd<'tcx, Normalize<T>>,
) -> Fallible<T>
where
    T: fmt::Debug + TypeFoldable<'tcx> + Lift<'gcx>,
{
    let (param_env, Normalize { value }) = key.into_parts();
    let Normalized { value, obligations } = infcx
        .at(&ObligationCause::dummy(), param_env)
        .normalize(&value)?;
    fulfill_cx.register_predicate_obligations(infcx, obligations);
    Ok(value)
}

fn type_op_normalize_ty(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<Ty<'tcx>>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResponse<'tcx, Ty<'tcx>>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_normalize_predicate(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<Predicate<'tcx>>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResponse<'tcx, Predicate<'tcx>>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_normalize_fn_sig(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<FnSig<'tcx>>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResponse<'tcx, FnSig<'tcx>>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_normalize_poly_fn_sig(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Normalize<PolyFnSig<'tcx>>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResponse<'tcx, PolyFnSig<'tcx>>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter_canonical_trait_query(&canonicalized, type_op_normalize)
}

fn type_op_subtype<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, Subtype<'tcx>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResponse<'tcx, ()>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter_canonical_trait_query(&canonicalized, |infcx, fulfill_cx, key| {
            let (param_env, Subtype { sub, sup }) = key.into_parts();
            Ok(infcx
                .at(&ObligationCause::dummy(), param_env)
                .sup(sup, sub)?
                .into_value_registering_obligations(infcx, fulfill_cx))
        })
}

fn type_op_prove_predicate<'tcx>(
    tcx: TyCtxt<'_, 'tcx, 'tcx>,
    canonicalized: Canonical<'tcx, ParamEnvAnd<'tcx, ProvePredicate<'tcx>>>,
) -> Result<Lrc<Canonical<'tcx, QueryResponse<'tcx, ()>>>, NoSolution> {
    tcx.infer_ctxt()
        .enter_canonical_trait_query(&canonicalized, |infcx, fulfill_cx, key| {
            let (param_env, ProvePredicate { predicate }) = key.into_parts();
            fulfill_cx.register_predicate_obligation(
                infcx,
                Obligation::new(ObligationCause::dummy(), param_env, predicate),
            );
            Ok(())
        })
}
