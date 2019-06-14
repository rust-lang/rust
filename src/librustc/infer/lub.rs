use super::combine::CombineFields;
use super::InferCtxt;
use super::lattice::{self, LatticeDir};
use super::Subtype;

use crate::traits::ObligationCause;
use crate::ty::{self, Ty, TyCtxt};
use crate::ty::relate::{Relate, RelateResult, TypeRelation};

/// "Least upper bound" (common supertype)
pub struct Lub<'combine, 'infcx, 'tcx> {
    fields: &'combine mut CombineFields<'infcx, 'tcx>,
    a_is_expected: bool,
}

impl<'combine, 'infcx, 'tcx> Lub<'combine, 'infcx, 'tcx> {
    pub fn new(
        fields: &'combine mut CombineFields<'infcx, 'tcx>,
        a_is_expected: bool,
    ) -> Lub<'combine, 'infcx, 'tcx> {
        Lub { fields: fields, a_is_expected: a_is_expected }
    }
}

impl TypeRelation<'tcx> for Lub<'combine, 'infcx, 'tcx> {
    fn tag(&self) -> &'static str { "Lub" }

    fn tcx(&self) -> TyCtxt<'tcx> { self.fields.tcx() }

    fn a_is_expected(&self) -> bool { self.a_is_expected }

    fn relate_with_variance<T: Relate<'tcx>>(&mut self,
                                             variance: ty::Variance,
                                             a: &T,
                                             b: &T)
                                             -> RelateResult<'tcx, T>
    {
        match variance {
            ty::Invariant => self.fields.equate(self.a_is_expected).relate(a, b),
            ty::Covariant => self.relate(a, b),
            // FIXME(#41044) -- not correct, need test
            ty::Bivariant => Ok(a.clone()),
            ty::Contravariant => self.fields.glb(self.a_is_expected).relate(a, b),
        }
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        lattice::super_lattice_tys(self, a, b)
    }

    fn regions(&mut self, a: ty::Region<'tcx>, b: ty::Region<'tcx>)
               -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug!("{}.regions({:?}, {:?})",
               self.tag(),
               a,
               b);

        let origin = Subtype(self.fields.trace.clone());
        Ok(self.fields.infcx.borrow_region_constraints().lub_regions(self.tcx(), origin, a, b))
    }

    fn consts(
        &mut self,
        a: &'tcx ty::Const<'tcx>,
        b: &'tcx ty::Const<'tcx>,
    ) -> RelateResult<'tcx, &'tcx ty::Const<'tcx>> {
        debug!("{}.consts({:?}, {:?})", self.tag(), a, b);
        if a == b {
            return Ok(a);
        }

        self.fields.infcx.super_combine_consts(self, a, b)
    }

    fn binders<T>(&mut self, a: &ty::Binder<T>, b: &ty::Binder<T>)
                  -> RelateResult<'tcx, ty::Binder<T>>
        where T: Relate<'tcx>
    {
        debug!("binders(a={:?}, b={:?})", a, b);

        // When higher-ranked types are involved, computing the LUB is
        // very challenging, switch to invariance. This is obviously
        // overly conservative but works ok in practice.
        self.relate_with_variance(ty::Variance::Invariant, a, b)?;
        Ok(a.clone())
    }
}

impl<'combine, 'infcx, 'tcx> LatticeDir<'infcx, 'tcx> for Lub<'combine, 'infcx, 'tcx> {
    fn infcx(&self) -> &'infcx InferCtxt<'infcx, 'tcx> {
        self.fields.infcx
    }

    fn cause(&self) -> &ObligationCause<'tcx> {
        &self.fields.trace.cause
    }

    fn relate_bound(&mut self, v: Ty<'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, ()> {
        let mut sub = self.fields.sub(self.a_is_expected);
        sub.relate(&a, &v)?;
        sub.relate(&b, &v)?;
        Ok(())
    }
}
