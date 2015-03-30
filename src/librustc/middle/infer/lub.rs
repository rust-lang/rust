// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::combine::*;
use super::higher_ranked::HigherRankedRelations;
use super::lattice::*;
use super::cres;
use super::Subtype;

use middle::ty::{self, Ty};
use util::ppaux::Repr;

/// "Least upper bound" (common supertype)
pub struct Lub<'f, 'tcx: 'f> {
    fields: CombineFields<'f, 'tcx>
}

#[allow(non_snake_case)]
pub fn Lub<'f, 'tcx>(cf: CombineFields<'f, 'tcx>) -> Lub<'f, 'tcx> {
    Lub { fields: cf }
}

impl<'f, 'tcx> Combine<'tcx> for Lub<'f, 'tcx> {
    fn tag(&self) -> String { "Lub".to_string() }
    fn fields<'a>(&'a self) -> &'a CombineFields<'a, 'tcx> { &self.fields }

    fn tys_with_variance(&self, v: ty::Variance, a: Ty<'tcx>, b: Ty<'tcx>)
                         -> cres<'tcx, Ty<'tcx>>
    {
        match v {
            ty::Invariant => self.equate().tys(a, b),
            ty::Covariant => self.tys(a, b),
            ty::Bivariant => self.bivariate().tys(a, b),
            ty::Contravariant => self.glb().tys(a, b),
        }
    }

    fn regions_with_variance(&self, v: ty::Variance, a: ty::Region, b: ty::Region)
                             -> cres<'tcx, ty::Region>
    {
        match v {
            ty::Invariant => self.equate().regions(a, b),
            ty::Covariant => self.regions(a, b),
            ty::Bivariant => self.bivariate().regions(a, b),
            ty::Contravariant => self.glb().regions(a, b),
        }
    }

    fn regions(&self, a: ty::Region, b: ty::Region) -> cres<'tcx, ty::Region> {
        debug!("{}.regions({}, {})",
               self.tag(),
               a.repr(self.tcx()),
               b.repr(self.tcx()));

        Ok(self.infcx().region_vars.lub_regions(Subtype(self.trace()), a, b))
    }

    fn tys(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> cres<'tcx, Ty<'tcx>> {
        super_lattice_tys(self, a, b)
    }

    fn binders<T>(&self, a: &ty::Binder<T>, b: &ty::Binder<T>) -> cres<'tcx, ty::Binder<T>>
        where T : Combineable<'tcx>
    {
        self.higher_ranked_lub(a, b)
    }
}
