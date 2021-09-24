use super::combine::{CombineFields, RelationDir};
use super::SubregionOrigin;

use crate::infer::combine::ConstEquateRelation;
use crate::traits::Obligation;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::relate::{Cause, Relate, RelateResult, TypeRelation};
use rustc_middle::ty::TyVar;
use rustc_middle::ty::{self, ToPredicate, Ty, TyCtxt};
use std::mem;

/// Ensures `a` is made a subtype of `b`. Returns `a` on success.
pub struct Sub<'combine, 'infcx, 'tcx> {
    fields: &'combine mut CombineFields<'infcx, 'tcx>,
    a_is_expected: bool,
}

impl<'combine, 'infcx, 'tcx> Sub<'combine, 'infcx, 'tcx> {
    pub fn new(
        f: &'combine mut CombineFields<'infcx, 'tcx>,
        a_is_expected: bool,
    ) -> Sub<'combine, 'infcx, 'tcx> {
        Sub { fields: f, a_is_expected }
    }

    fn with_expected_switched<R, F: FnOnce(&mut Self) -> R>(&mut self, f: F) -> R {
        self.a_is_expected = !self.a_is_expected;
        let result = f(self);
        self.a_is_expected = !self.a_is_expected;
        result
    }
}

impl TypeRelation<'tcx> for Sub<'combine, 'infcx, 'tcx> {
    fn tag(&self) -> &'static str {
        "Sub"
    }
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.fields.infcx.tcx
    }

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.fields.param_env
    }

    fn a_is_expected(&self) -> bool {
        self.a_is_expected
    }

    fn with_cause<F, R>(&mut self, cause: Cause, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        debug!("sub with_cause={:?}", cause);
        let old_cause = mem::replace(&mut self.fields.cause, Some(cause));
        let r = f(self);
        debug!("sub old_cause={:?}", old_cause);
        self.fields.cause = old_cause;
        r
    }

    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        variance: ty::Variance,
        _info: ty::VarianceDiagInfo<'tcx>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        match variance {
            ty::Invariant => self.fields.equate(self.a_is_expected).relate(a, b),
            ty::Covariant => self.relate(a, b),
            ty::Bivariant => Ok(a),
            ty::Contravariant => self.with_expected_switched(|this| this.relate(b, a)),
        }
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        debug!("{}.tys({:?}, {:?})", self.tag(), a, b);

        if a == b {
            return Ok(a);
        }

        let infcx = self.fields.infcx;
        let a = infcx.inner.borrow_mut().type_variables().replace_if_possible(a);
        let b = infcx.inner.borrow_mut().type_variables().replace_if_possible(b);
        match (a.kind(), b.kind()) {
            (&ty::Infer(TyVar(_)), &ty::Infer(TyVar(_))) => {
                // Shouldn't have any LBR here, so we can safely put
                // this under a binder below without fear of accidental
                // capture.
                assert!(!a.has_escaping_bound_vars());
                assert!(!b.has_escaping_bound_vars());

                // can't make progress on `A <: B` if both A and B are
                // type variables, so record an obligation.
                self.fields.obligations.push(Obligation::new(
                    self.fields.trace.cause.clone(),
                    self.fields.param_env,
                    ty::Binder::dummy(ty::PredicateKind::Subtype(ty::SubtypePredicate {
                        a_is_expected: self.a_is_expected,
                        a,
                        b,
                    }))
                    .to_predicate(self.tcx()),
                ));

                Ok(a)
            }
            (&ty::Infer(TyVar(a_id)), _) => {
                self.fields.instantiate(b, RelationDir::SupertypeOf, a_id, !self.a_is_expected)?;
                Ok(a)
            }
            (_, &ty::Infer(TyVar(b_id))) => {
                self.fields.instantiate(a, RelationDir::SubtypeOf, b_id, self.a_is_expected)?;
                Ok(a)
            }

            (&ty::Error(_), _) | (_, &ty::Error(_)) => {
                infcx.set_tainted_by_errors();
                Ok(self.tcx().ty_error())
            }

            _ => {
                self.fields.infcx.super_combine_tys(self, a, b)?;
                Ok(a)
            }
        }
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug!("{}.regions({:?}, {:?}) self.cause={:?}", self.tag(), a, b, self.fields.cause);

        // FIXME -- we have more fine-grained information available
        // from the "cause" field, we could perhaps give more tailored
        // error messages.
        let origin = SubregionOrigin::Subtype(Box::new(self.fields.trace.clone()));
        self.fields
            .infcx
            .inner
            .borrow_mut()
            .unwrap_region_constraints()
            .make_subregion(origin, a, b);

        Ok(a)
    }

    fn consts(
        &mut self,
        a: &'tcx ty::Const<'tcx>,
        b: &'tcx ty::Const<'tcx>,
    ) -> RelateResult<'tcx, &'tcx ty::Const<'tcx>> {
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
        self.fields.higher_ranked_sub(a, b, self.a_is_expected)
    }
}

impl<'tcx> ConstEquateRelation<'tcx> for Sub<'_, '_, 'tcx> {
    fn const_equate_obligation(&mut self, a: &'tcx ty::Const<'tcx>, b: &'tcx ty::Const<'tcx>) {
        self.fields.add_const_equate_obligation(self.a_is_expected, a, b);
    }
}
