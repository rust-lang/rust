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

use ty::{self, Ty, TyCtxt};
use ty::relate::{Relate, RelateResult, TypeRelation};
use traits::PredicateObligations;

/// "Greatest lower bound" (common subtype)
pub struct Glb<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    fields: CombineFields<'a, 'gcx, 'tcx>
}

impl<'a, 'gcx, 'tcx> Glb<'a, 'gcx, 'tcx> {
    pub fn new(fields: CombineFields<'a, 'gcx, 'tcx>) -> Glb<'a, 'gcx, 'tcx> {
        Glb { fields: fields }
    }

    pub fn obligations(self) -> PredicateObligations<'tcx> {
        self.fields.obligations
    }
}

impl<'a, 'gcx, 'tcx> TypeRelation<'a, 'gcx, 'tcx> for Glb<'a, 'gcx, 'tcx> {
    fn tag(&self) -> &'static str { "Glb" }

    fn tcx(&self) -> TyCtxt<'a, 'gcx, 'tcx> { self.fields.tcx() }

    fn a_is_expected(&self) -> bool { self.fields.a_is_expected }

    fn relate_with_variance<T: Relate<'tcx>>(&mut self,
                                             variance: ty::Variance,
                                             a: &T,
                                             b: &T)
                                             -> RelateResult<'tcx, T>
    {
        match variance {
            ty::Invariant => self.fields.equate().relate(a, b),
            ty::Covariant => self.relate(a, b),
            ty::Bivariant => self.fields.bivariate().relate(a, b),
            ty::Contravariant => self.fields.lub().relate(a, b),
        }
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        lattice::super_lattice_tys(self, a, b)
    }

    fn regions(&mut self, a: ty::Region, b: ty::Region) -> RelateResult<'tcx, ty::Region> {
        debug!("{}.regions({:?}, {:?})",
               self.tag(),
               a,
               b);

        let origin = Subtype(self.fields.trace.clone());
        Ok(self.fields.infcx.region_vars.glb_regions(origin, a, b))
    }

    fn binders<T>(&mut self, a: &ty::Binder<T>, b: &ty::Binder<T>)
                  -> RelateResult<'tcx, ty::Binder<T>>
        where T: Relate<'tcx>
    {
        self.fields.higher_ranked_glb(a, b)
    }
}

impl<'a, 'gcx, 'tcx> LatticeDir<'a, 'gcx, 'tcx> for Glb<'a, 'gcx, 'tcx> {
    fn infcx(&self) -> &'a InferCtxt<'a, 'gcx, 'tcx> {
        self.fields.infcx
    }

    fn relate_bound(&self, v: Ty<'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, ()> {
        let mut sub = self.fields.sub();
        sub.relate(&v, &a)?;
        sub.relate(&v, &b)?;
        Ok(())
    }
}
