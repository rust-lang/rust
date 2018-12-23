// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::combine::CombineFields;
use super::InferCtxt;
use super::lattice::{self, LatticeDir};
use super::Subtype;

use traits::ObligationCause;
use ty::{self, Ty, TyCtxt};
use ty::relate::{self, Relate, RelateResult, TypeRelation};

/// "Greatest lower bound" (common subtype)
pub struct Glb<'combine, 'infcx: 'combine, 'gcx: 'infcx+'tcx, 'tcx: 'infcx> {
    fields: &'combine mut CombineFields<'infcx, 'gcx, 'tcx>,
    a_is_expected: bool,
}

impl<'combine, 'infcx, 'gcx, 'tcx> Glb<'combine, 'infcx, 'gcx, 'tcx> {
    pub fn new(fields: &'combine mut CombineFields<'infcx, 'gcx, 'tcx>, a_is_expected: bool)
        -> Glb<'combine, 'infcx, 'gcx, 'tcx>
    {
        Glb { fields: fields, a_is_expected: a_is_expected }
    }
}

impl<'combine, 'infcx, 'gcx, 'tcx> TypeRelation<'infcx, 'gcx, 'tcx>
    for Glb<'combine, 'infcx, 'gcx, 'tcx>
{
    fn tag(&self) -> &'static str { "Glb" }

    fn trait_object_mode(&self) -> relate::TraitObjectMode {
        self.fields.infcx.trait_object_mode()
    }

    fn tcx(&self) -> TyCtxt<'infcx, 'gcx, 'tcx> { self.fields.tcx() }

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
            ty::Contravariant => self.fields.lub(self.a_is_expected).relate(a, b),
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
        Ok(self.fields.infcx.borrow_region_constraints().glb_regions(self.tcx(), origin, a, b))
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

impl<'combine, 'infcx, 'gcx, 'tcx> LatticeDir<'infcx, 'gcx, 'tcx>
    for Glb<'combine, 'infcx, 'gcx, 'tcx>
{
    fn infcx(&self) -> &'infcx InferCtxt<'infcx, 'gcx, 'tcx> {
        self.fields.infcx
    }

    fn cause(&self) -> &ObligationCause<'tcx> {
        &self.fields.trace.cause
    }

    fn relate_bound(&mut self, v: Ty<'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, ()> {
        let mut sub = self.fields.sub(self.a_is_expected);
        sub.relate(&v, &a)?;
        sub.relate(&v, &b)?;
        Ok(())
    }
}
