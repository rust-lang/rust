use crate::traits::PredicateObligations;

use super::combine::{CombineFields, ObligationEmittingRelation, RelationDir};
use super::Subtype;

use rustc_middle::ty::relate::{self, Relate, RelateResult, TypeRelation};
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::TyVar;
use rustc_middle::ty::{self, Ty, TyCtxt};

use rustc_hir::def_id::DefId;

/// Ensures `a` is made equal to `b`. Returns `a` on success.
pub struct Equate<'combine, 'infcx, 'tcx> {
    fields: &'combine mut CombineFields<'infcx, 'tcx>,
    a_is_expected: bool,
}

impl<'combine, 'infcx, 'tcx> Equate<'combine, 'infcx, 'tcx> {
    pub fn new(
        fields: &'combine mut CombineFields<'infcx, 'tcx>,
        a_is_expected: bool,
    ) -> Equate<'combine, 'infcx, 'tcx> {
        Equate { fields, a_is_expected }
    }
}

impl<'tcx> TypeRelation<'tcx> for Equate<'_, '_, 'tcx> {
    fn tag(&self) -> &'static str {
        "Equate"
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.fields.tcx()
    }

    fn intercrate(&self) -> bool {
        self.fields.infcx.intercrate
    }

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.fields.param_env
    }

    fn a_is_expected(&self) -> bool {
        self.a_is_expected
    }

    fn mark_ambiguous(&mut self) {
        self.fields.mark_ambiguous();
    }

    fn relate_item_substs(
        &mut self,
        _item_def_id: DefId,
        a_subst: SubstsRef<'tcx>,
        b_subst: SubstsRef<'tcx>,
    ) -> RelateResult<'tcx, SubstsRef<'tcx>> {
        // N.B., once we are equating types, we don't care about
        // variance, so don't try to lookup the variance here. This
        // also avoids some cycles (e.g., #41849) since looking up
        // variance requires computing types which can require
        // performing trait matching (which then performs equality
        // unification).

        relate::relate_substs(self, a_subst, b_subst)
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

            (&ty::Infer(TyVar(a_id)), _) => {
                self.fields.instantiate(b, RelationDir::EqTo, a_id, self.a_is_expected)?;
            }

            (_, &ty::Infer(TyVar(b_id))) => {
                self.fields.instantiate(a, RelationDir::EqTo, b_id, self.a_is_expected)?;
            }

            (
                &ty::Alias(ty::Opaque, ty::AliasTy { def_id: a_def_id, .. }),
                &ty::Alias(ty::Opaque, ty::AliasTy { def_id: b_def_id, .. }),
            ) if a_def_id == b_def_id => {
                self.fields.infcx.super_combine_tys(self, a, b)?;
            }
            (&ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }), _)
            | (_, &ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }))
                if self.fields.define_opaque_types && def_id.is_local() =>
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
            // Optimization of GeneratorWitness relation since we know that all
            // free regions are replaced with bound regions during construction.
            // This greatly speeds up equating of GeneratorWitness.
            (&ty::GeneratorWitness(a_types), &ty::GeneratorWitness(b_types)) => {
                let a_types = infcx.tcx.anonymize_bound_vars(a_types);
                let b_types = infcx.tcx.anonymize_bound_vars(b_types);
                if a_types.bound_vars() == b_types.bound_vars() {
                    let (a_types, b_types) = infcx.instantiate_binder_with_placeholders(
                        a_types.map_bound(|a_types| (a_types, b_types.skip_binder())),
                    );
                    for (a, b) in std::iter::zip(a_types, b_types) {
                        self.relate(a, b)?;
                    }
                } else {
                    return Err(ty::error::TypeError::Sorts(ty::relate::expected_found(
                        self, a, b,
                    )));
                }
            }

            _ => {
                self.fields.infcx.super_combine_tys(self, a, b)?;
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
        let origin = Subtype(Box::new(self.fields.trace.clone()));
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
        // A binder is equal to itself if it's structually equal to itself
        if a == b {
            return Ok(a);
        }

        if a.skip_binder().has_escaping_bound_vars() || b.skip_binder().has_escaping_bound_vars() {
            self.fields.higher_ranked_sub(a, b, self.a_is_expected)?;
            self.fields.higher_ranked_sub(b, a, self.a_is_expected)?;
        } else {
            // Fast path for the common case.
            self.relate(a.skip_binder(), b.skip_binder())?;
        }
        Ok(a)
    }
}

impl<'tcx> ObligationEmittingRelation<'tcx> for Equate<'_, '_, 'tcx> {
    fn register_predicates(
        &mut self,
        obligations: impl IntoIterator<Item = impl ty::ToPredicate<'tcx>>,
    ) {
        self.fields.register_predicates(obligations);
    }

    fn register_obligations(&mut self, obligations: PredicateObligations<'tcx>) {
        self.fields.register_obligations(obligations);
    }
}
