// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use transform::nll::ToRegionVid;
use rustc::ty::{self, Ty, TyCtxt, RegionVid};
use rustc::ty::relate::{self, Relate, RelateResult, TypeRelation};

pub fn outlives_pairs<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                      a: Ty<'tcx>,
                      b: Ty<'tcx>)
                      -> Vec<(RegionVid, RegionVid)>
{
    let mut subtype = Subtype::new(tcx);
    match subtype.relate(&a, &b) {
        Ok(_) => subtype.outlives_pairs,

        Err(_) => bug!("Fail to relate a = {:?} and b = {:?}", a, b)
    }
}

struct Subtype<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    outlives_pairs: Vec<(RegionVid, RegionVid)>,
    ambient_variance: ty::Variance,
}

impl<'a, 'gcx, 'tcx> Subtype<'a, 'gcx, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Subtype<'a, 'gcx, 'tcx> {
        Subtype {
            tcx,
            outlives_pairs: vec![],
            ambient_variance: ty::Covariant,
        }
    }
}

impl<'a, 'gcx, 'tcx> TypeRelation<'a, 'gcx, 'tcx> for Subtype<'a, 'gcx, 'tcx> {
    fn tag(&self) -> &'static str { "Subtype" }
    fn tcx(&self) -> TyCtxt<'a, 'gcx, 'tcx> { self.tcx }
    fn a_is_expected(&self) -> bool { true } // irrelevant

    fn relate_with_variance<T: Relate<'tcx>>(&mut self,
                                             variance: ty::Variance,
                                             a: &T,
                                             b: &T)
                                             -> RelateResult<'tcx, T>
    {
        let old_ambient_variance = self.ambient_variance;
        self.ambient_variance = self.ambient_variance.xform(variance);

        let result = self.relate(a, b);
        self.ambient_variance = old_ambient_variance;
        result
    }

    fn tys(&mut self, t: Ty<'tcx>, t2: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        relate::super_relate_tys(self, t, t2)
    }

    fn regions(&mut self, r_a: ty::Region<'tcx>, r_b: ty::Region<'tcx>)
               -> RelateResult<'tcx, ty::Region<'tcx>> {
        let a = r_a.to_region_vid();
        let b = r_b.to_region_vid();

        match self.ambient_variance {
            ty::Covariant => {
                self.outlives_pairs.push((b, a));
            },

            ty::Invariant => {
                self.outlives_pairs.push((a, b));
                self.outlives_pairs.push((b, a));
            },

            ty::Contravariant => {
                self.outlives_pairs.push((a, b));
            },

            ty::Bivariant => {},
        }

        Ok(r_a)
    }

    fn binders<T>(&mut self, _a: &ty::Binder<T>, _b: &ty::Binder<T>)
                  -> RelateResult<'tcx, ty::Binder<T>>
        where T: Relate<'tcx>
    {
        unimplemented!();
    }
}
