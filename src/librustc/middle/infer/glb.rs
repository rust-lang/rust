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
use super::lattice::*;
use super::higher_ranked::HigherRankedRelations;
use super::{cres};
use super::Subtype;

use middle::ty::{self, Ty};
use syntax::ast::{MutImmutable, MutMutable, Unsafety};
use util::ppaux::mt_to_string;
use util::ppaux::Repr;

/// "Greatest lower bound" (common subtype)
pub struct Glb<'f, 'tcx: 'f> {
    fields: CombineFields<'f, 'tcx>
}

#[allow(non_snake_case)]
pub fn Glb<'f, 'tcx>(cf: CombineFields<'f, 'tcx>) -> Glb<'f, 'tcx> {
    Glb { fields: cf }
}

impl<'f, 'tcx> Combine<'tcx> for Glb<'f, 'tcx> {
    fn tag(&self) -> String { "Glb".to_string() }
    fn fields<'a>(&'a self) -> &'a CombineFields<'a, 'tcx> { &self.fields }

    fn tys_with_variance(&self, v: ty::Variance, a: Ty<'tcx>, b: Ty<'tcx>)
                         -> cres<'tcx, Ty<'tcx>>
    {
        match v {
            ty::Invariant => self.equate().tys(a, b),
            ty::Covariant => self.tys(a, b),
            ty::Bivariant => self.bivariate().tys(a, b),
            ty::Contravariant => self.lub().tys(a, b),
        }
    }

    fn regions_with_variance(&self, v: ty::Variance, a: ty::Region, b: ty::Region)
                             -> cres<'tcx, ty::Region>
    {
        match v {
            ty::Invariant => self.equate().regions(a, b),
            ty::Covariant => self.regions(a, b),
            ty::Bivariant => self.bivariate().regions(a, b),
            ty::Contravariant => self.lub().regions(a, b),
        }
    }


    fn unsafeties(&self, a: Unsafety, b: Unsafety) -> cres<'tcx, Unsafety> {
        match (a, b) {
          (Unsafety::Normal, _) | (_, Unsafety::Normal) => Ok(Unsafety::Normal),
          (Unsafety::Unsafe, Unsafety::Unsafe) => Ok(Unsafety::Unsafe)
        }
    }

    fn regions(&self, a: ty::Region, b: ty::Region) -> cres<'tcx, ty::Region> {
        debug!("{}.regions({}, {})",
               self.tag(),
               a.repr(self.fields.infcx.tcx),
               b.repr(self.fields.infcx.tcx));

        Ok(self.fields.infcx.region_vars.glb_regions(Subtype(self.trace()), a, b))
    }

    fn tys(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> cres<'tcx, Ty<'tcx>> {
        super_lattice_tys(self, a, b)
    }

    fn binders<T>(&self, a: &ty::Binder<T>, b: &ty::Binder<T>) -> cres<'tcx, ty::Binder<T>>
        where T : Combineable<'tcx>
    {
        self.higher_ranked_glb(a, b)
    }
}
