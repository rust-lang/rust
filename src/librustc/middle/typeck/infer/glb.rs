// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use middle::ty::{BuiltinBounds};
use middle::ty::{mod, Ty};
use middle::typeck::infer::combine::*;
use middle::typeck::infer::lattice::*;
use middle::typeck::infer::equate::Equate;
use middle::typeck::infer::higher_ranked::HigherRankedRelations;
use middle::typeck::infer::lub::Lub;
use middle::typeck::infer::sub::Sub;
use middle::typeck::infer::{cres, InferCtxt};
use middle::typeck::infer::{TypeTrace, Subtype};
use syntax::ast::{Many, Once, MutImmutable, MutMutable};
use syntax::ast::{NormalFn, UnsafeFn};
use syntax::ast::{Onceness, FnStyle};
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
    fn infcx<'a>(&'a self) -> &'a InferCtxt<'a, 'tcx> { self.fields.infcx }
    fn tag(&self) -> String { "glb".to_string() }
    fn a_is_expected(&self) -> bool { self.fields.a_is_expected }
    fn trace(&self) -> TypeTrace<'tcx> { self.fields.trace.clone() }

    fn equate<'a>(&'a self) -> Equate<'a, 'tcx> { Equate(self.fields.clone()) }
    fn sub<'a>(&'a self) -> Sub<'a, 'tcx> { Sub(self.fields.clone()) }
    fn lub<'a>(&'a self) -> Lub<'a, 'tcx> { Lub(self.fields.clone()) }
    fn glb<'a>(&'a self) -> Glb<'a, 'tcx> { Glb(self.fields.clone()) }

    fn mts(&self, a: &ty::mt<'tcx>, b: &ty::mt<'tcx>) -> cres<'tcx, ty::mt<'tcx>> {
        let tcx = self.fields.infcx.tcx;

        debug!("{}.mts({}, {})",
               self.tag(),
               mt_to_string(tcx, a),
               mt_to_string(tcx, b));

        match (a.mutbl, b.mutbl) {
            // If one side or both is mut, then the GLB must use
            // the precise type from the mut side.
            (MutMutable, MutMutable) => {
                let t = try!(self.equate().tys(a.ty, b.ty));
                Ok(ty::mt {ty: t, mutbl: MutMutable})
            }

            // If one side or both is immutable, we can use the GLB of
            // both sides but mutbl must be `MutImmutable`.
            (MutImmutable, MutImmutable) => {
                let t = try!(self.tys(a.ty, b.ty));
                Ok(ty::mt {ty: t, mutbl: MutImmutable})
            }

            // There is no mutual subtype of these combinations.
            (MutMutable, MutImmutable) |
            (MutImmutable, MutMutable) => {
                Err(ty::terr_mutability)
            }
        }
    }

    fn contratys(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> cres<'tcx, Ty<'tcx>> {
        self.lub().tys(a, b)
    }

    fn fn_styles(&self, a: FnStyle, b: FnStyle) -> cres<'tcx, FnStyle> {
        match (a, b) {
          (NormalFn, _) | (_, NormalFn) => Ok(NormalFn),
          (UnsafeFn, UnsafeFn) => Ok(UnsafeFn)
        }
    }

    fn oncenesses(&self, a: Onceness, b: Onceness) -> cres<'tcx, Onceness> {
        match (a, b) {
            (Many, _) | (_, Many) => Ok(Many),
            (Once, Once) => Ok(Once)
        }
    }

    fn builtin_bounds(&self,
                      a: ty::BuiltinBounds,
                      b: ty::BuiltinBounds)
                      -> cres<'tcx, ty::BuiltinBounds> {
        // More bounds is a subtype of fewer bounds, so
        // the GLB (mutual subtype) is the union.
        Ok(a.union(b))
    }

    fn regions(&self, a: ty::Region, b: ty::Region) -> cres<'tcx, ty::Region> {
        debug!("{}.regions({}, {})",
               self.tag(),
               a.repr(self.fields.infcx.tcx),
               b.repr(self.fields.infcx.tcx));

        Ok(self.fields.infcx.region_vars.glb_regions(Subtype(self.trace()), a, b))
    }

    fn contraregions(&self, a: ty::Region, b: ty::Region)
                    -> cres<'tcx, ty::Region> {
        self.lub().regions(a, b)
    }

    fn tys(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> cres<'tcx, Ty<'tcx>> {
        super_lattice_tys(self, a, b)
    }

    fn fn_sigs(&self, a: &ty::FnSig<'tcx>, b: &ty::FnSig<'tcx>)
               -> cres<'tcx, ty::FnSig<'tcx>> {
        self.higher_ranked_glb(a, b)
    }

    fn trait_refs(&self, a: &ty::TraitRef<'tcx>, b: &ty::TraitRef<'tcx>)
                  -> cres<'tcx, ty::TraitRef<'tcx>> {
        self.higher_ranked_glb(a, b)
    }
}
