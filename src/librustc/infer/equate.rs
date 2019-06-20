use super::combine::{CombineFields, RelationDir, const_unification_error};
use super::Subtype;

use crate::hir::def_id::DefId;

use crate::ty::{self, Ty, TyCtxt, InferConst};
use crate::ty::TyVar;
use crate::ty::subst::SubstsRef;
use crate::ty::relate::{self, Relate, RelateResult, TypeRelation};
use crate::mir::interpret::ConstValue;
use crate::infer::unify_key::replace_if_possible;

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
        Equate { fields: fields, a_is_expected: a_is_expected }
    }
}

impl TypeRelation<'tcx> for Equate<'combine, 'infcx, 'tcx> {
    fn tag(&self) -> &'static str { "Equate" }

    fn tcx(&self) -> TyCtxt<'tcx> { self.fields.tcx() }

    fn a_is_expected(&self) -> bool { self.a_is_expected }

    fn relate_item_substs(&mut self,
                          _item_def_id: DefId,
                          a_subst: SubstsRef<'tcx>,
                          b_subst: SubstsRef<'tcx>)
                          -> RelateResult<'tcx, SubstsRef<'tcx>>
    {
        // N.B., once we are equating types, we don't care about
        // variance, so don't try to lookup the variance here. This
        // also avoids some cycles (e.g., #41849) since looking up
        // variance requires computing types which can require
        // performing trait matching (which then performs equality
        // unification).

        relate::relate_substs(self, None, a_subst, b_subst)
    }

    fn relate_with_variance<T: Relate<'tcx>>(&mut self,
                                             _: ty::Variance,
                                             a: &T,
                                             b: &T)
                                             -> RelateResult<'tcx, T>
    {
        self.relate(a, b)
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        debug!("{}.tys({:?}, {:?})", self.tag(),
               a, b);
        if a == b { return Ok(a); }

        let infcx = self.fields.infcx;
        let a = infcx.type_variables.borrow_mut().replace_if_possible(a);
        let b = infcx.type_variables.borrow_mut().replace_if_possible(b);

        debug!("{}.tys: replacements ({:?}, {:?})", self.tag(), a, b);

        match (&a.sty, &b.sty) {
            (&ty::Infer(TyVar(a_id)), &ty::Infer(TyVar(b_id))) => {
                infcx.type_variables.borrow_mut().equate(a_id, b_id);
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

    fn regions(&mut self, a: ty::Region<'tcx>, b: ty::Region<'tcx>)
               -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug!("{}.regions({:?}, {:?})",
               self.tag(),
               a,
               b);
        let origin = Subtype(self.fields.trace.clone());
        self.fields.infcx.borrow_region_constraints()
                         .make_eqregion(origin, a, b);
        Ok(a)
    }

    fn consts(
        &mut self,
        a: &'tcx ty::Const<'tcx>,
        b: &'tcx ty::Const<'tcx>,
    ) -> RelateResult<'tcx, &'tcx ty::Const<'tcx>> {
        debug!("{}.consts({:?}, {:?})", self.tag(), a, b);
        if a == b { return Ok(a); }

        let infcx = self.fields.infcx;
        let a = replace_if_possible(infcx.const_unification_table.borrow_mut(), a);
        let b = replace_if_possible(infcx.const_unification_table.borrow_mut(), b);
        let a_is_expected = self.a_is_expected();

        match (a.val, b.val) {
            (ConstValue::Infer(InferConst::Var(a_vid)),
                ConstValue::Infer(InferConst::Var(b_vid))) => {
                infcx.const_unification_table
                    .borrow_mut()
                    .unify_var_var(a_vid, b_vid)
                    .map_err(|e| const_unification_error(a_is_expected, e))?;
                return Ok(a);
            }

            (ConstValue::Infer(InferConst::Var(a_id)), _) => {
                self.fields.infcx.unify_const_variable(a_is_expected, a_id, b)?;
                return Ok(a);
            }

            (_, ConstValue::Infer(InferConst::Var(b_id))) => {
                self.fields.infcx.unify_const_variable(!a_is_expected, b_id, a)?;
                return Ok(a);
            }

            _ => {}
        }

        self.fields.infcx.super_combine_consts(self, a, b)?;
        Ok(a)
    }

    fn binders<T>(&mut self, a: &ty::Binder<T>, b: &ty::Binder<T>)
                  -> RelateResult<'tcx, ty::Binder<T>>
        where T: Relate<'tcx>
    {
        self.fields.higher_ranked_sub(a, b, self.a_is_expected)?;
        self.fields.higher_ranked_sub(b, a, self.a_is_expected)
    }
}
