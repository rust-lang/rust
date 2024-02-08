use super::combine::{CombineFields, ObligationEmittingRelation};
use super::StructurallyRelateAliases;
use crate::infer::BoundRegionConversionTime::HigherRankedType;
use crate::infer::{DefineOpaqueTypes, SubregionOrigin};
use crate::traits::PredicateObligations;

use rustc_middle::ty::relate::{self, Relate, RelateResult, TypeRelation};
use rustc_middle::ty::GenericArgsRef;
use rustc_middle::ty::TyVar;
use rustc_middle::ty::{self, Ty, TyCtxt};

use rustc_hir::def_id::DefId;
use rustc_span::Span;

/// Ensures `a` is made equal to `b`. Returns `a` on success.
pub struct Equate<'combine, 'infcx, 'tcx> {
    fields: &'combine mut CombineFields<'infcx, 'tcx>,
    structurally_relate_aliases: StructurallyRelateAliases,
    a_is_expected: bool,
}

impl<'combine, 'infcx, 'tcx> Equate<'combine, 'infcx, 'tcx> {
    pub fn new(
        fields: &'combine mut CombineFields<'infcx, 'tcx>,
        structurally_relate_aliases: StructurallyRelateAliases,
        a_is_expected: bool,
    ) -> Equate<'combine, 'infcx, 'tcx> {
        Equate { fields, structurally_relate_aliases, a_is_expected }
    }
}

impl<'tcx> TypeRelation<'tcx> for Equate<'_, '_, 'tcx> {
    fn tag(&self) -> &'static str {
        "Equate"
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.fields.tcx()
    }

    fn a_is_expected(&self) -> bool {
        self.a_is_expected
    }

    fn relate_item_args(
        &mut self,
        _item_def_id: DefId,
        a_arg: GenericArgsRef<'tcx>,
        b_arg: GenericArgsRef<'tcx>,
    ) -> RelateResult<'tcx, GenericArgsRef<'tcx>> {
        // N.B., once we are equating types, we don't care about
        // variance, so don't try to lookup the variance here. This
        // also avoids some cycles (e.g., #41849) since looking up
        // variance requires computing types which can require
        // performing trait matching (which then performs equality
        // unification).

        relate::relate_args_invariantly(self, a_arg, b_arg)
    }

    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        _: ty::Variance,
        _info: ty::VarianceDiagInfo<'tcx>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        self.relate(a, b)
    }

    #[instrument(skip(self), level = "debug")]
    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        if a == b {
            return Ok(a);
        }

        trace!(a = ?a.kind(), b = ?b.kind());

        let infcx = self.fields.infcx;

        let a = infcx.inner.borrow_mut().type_variables().replace_if_possible(a);
        let b = infcx.inner.borrow_mut().type_variables().replace_if_possible(b);

        match (a.kind(), b.kind()) {
            (&ty::Infer(TyVar(a_id)), &ty::Infer(TyVar(b_id))) => {
                infcx.inner.borrow_mut().type_variables().equate(a_id, b_id);
            }

            (&ty::Infer(TyVar(a_vid)), _) => {
                infcx.instantiate_ty_var(self, self.a_is_expected, a_vid, ty::Invariant, b)?;
            }

            (_, &ty::Infer(TyVar(b_vid))) => {
                infcx.instantiate_ty_var(self, !self.a_is_expected, b_vid, ty::Invariant, a)?;
            }

            (&ty::Error(e), _) | (_, &ty::Error(e)) => {
                infcx.set_tainted_by_errors(e);
                return Ok(Ty::new_error(self.tcx(), e));
            }

            (
                &ty::Alias(ty::Opaque, ty::AliasTy { def_id: a_def_id, .. }),
                &ty::Alias(ty::Opaque, ty::AliasTy { def_id: b_def_id, .. }),
            ) if a_def_id == b_def_id => {
                infcx.super_combine_tys(self, a, b)?;
            }
            (&ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }), _)
            | (_, &ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }))
                if self.fields.define_opaque_types == DefineOpaqueTypes::Yes
                    && def_id.is_local()
                    && !self.fields.infcx.next_trait_solver() =>
            {
                self.fields.obligations.extend(
                    infcx
                        .handle_opaque_type(
                            a,
                            b,
                            self.a_is_expected(),
                            &self.fields.trace.cause,
                            self.param_env(),
                        )?
                        .obligations,
                );
            }
            _ => {
                infcx.super_combine_tys(self, a, b)?;
            }
        }

        Ok(a)
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug!("{}.regions({:?}, {:?})", self.tag(), a, b);
        let origin = SubregionOrigin::Subtype(Box::new(self.fields.trace.clone()));
        self.fields
            .infcx
            .inner
            .borrow_mut()
            .unwrap_region_constraints()
            .make_eqregion(origin, a, b);
        Ok(a)
    }

    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        self.fields.infcx.super_combine_consts(self, a, b)
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<'tcx>,
    {
        // A binder is equal to itself if it's structurally equal to itself
        if a == b {
            return Ok(a);
        }

        if let (Some(a), Some(b)) = (a.no_bound_vars(), b.no_bound_vars()) {
            // Fast path for the common case.
            self.relate(a, b)?;
        } else {
            // When equating binders, we check that there is a 1-to-1
            // correspondence between the bound vars in both types.
            //
            // We do so by separately instantiating one of the binders with
            // placeholders and the other with inference variables and then
            // equating the instantiated types.
            //
            // We want `for<..> A == for<..> B` -- therefore we want
            // `exists<..> A == for<..> B` and `exists<..> B == for<..> A`.

            let span = self.fields.trace.cause.span;
            let infcx = self.fields.infcx;

            // Check if `exists<..> A == for<..> B`
            infcx.enter_forall(b, |b| {
                let a = infcx.instantiate_binder_with_fresh_vars(span, HigherRankedType, a);
                self.relate(a, b)
            })?;

            // Check if `exists<..> B == for<..> A`.
            infcx.enter_forall(a, |a| {
                let b = infcx.instantiate_binder_with_fresh_vars(span, HigherRankedType, b);
                self.relate(a, b)
            })?;
        }
        Ok(a)
    }
}

impl<'tcx> ObligationEmittingRelation<'tcx> for Equate<'_, '_, 'tcx> {
    fn span(&self) -> Span {
        self.fields.trace.span()
    }

    fn structurally_relate_aliases(&self) -> StructurallyRelateAliases {
        self.structurally_relate_aliases
    }

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.fields.param_env
    }

    fn register_predicates(&mut self, obligations: impl IntoIterator<Item: ty::ToPredicate<'tcx>>) {
        self.fields.register_predicates(obligations);
    }

    fn register_obligations(&mut self, obligations: PredicateObligations<'tcx>) {
        self.fields.register_obligations(obligations);
    }

    fn alias_relate_direction(&self) -> ty::AliasRelationDirection {
        ty::AliasRelationDirection::Equate
    }
}
