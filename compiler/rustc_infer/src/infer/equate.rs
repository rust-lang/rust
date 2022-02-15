use super::combine::{CombineFields, ConstEquateRelation, RelationDir};
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

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.fields.param_env
    }

    fn a_is_expected(&self) -> bool {
        self.a_is_expected
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

        relate::relate_substs(self, None, a_subst, b_subst)
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

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        debug!("{}.tys({:?}, {:?})", self.tag(), a, b);
        if a == b {
            return Ok(a);
        }

        let infcx = self.fields.infcx;
        let a = infcx.inner.borrow_mut().type_variables().replace_if_possible(a);
        let b = infcx.inner.borrow_mut().type_variables().replace_if_possible(b);

        debug!("{}.tys: replacements ({:?}, {:?})", self.tag(), a, b);

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
        if a.skip_binder().has_escaping_bound_vars() || b.skip_binder().has_escaping_bound_vars() {
            self.fields.higher_ranked_sub(a, b, self.a_is_expected)?;
            self.fields.higher_ranked_sub(b, a, self.a_is_expected)
        } else {
            // Fast path for the common case.
            self.relate(a.skip_binder(), b.skip_binder())?;
            Ok(a)
        }
    }
}

impl<'tcx> ConstEquateRelation<'tcx> for Equate<'_, '_, 'tcx> {
    fn const_equate_obligation(&mut self, a: ty::Const<'tcx>, b: ty::Const<'tcx>) {
        self.fields.add_const_equate_obligation(self.a_is_expected, a, b);
    }
}
