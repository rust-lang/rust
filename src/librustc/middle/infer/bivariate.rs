// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Applies the "bivariance relationship" to two types and/or regions.
//! If (A,B) are bivariant then either A <: B or B <: A. It occurs
//! when type/lifetime parameters are unconstrained. Usually this is
//! an error, but we permit it in the specific case where a type
//! parameter is constrained in a where-clause via an associated type.
//!
//! There are several ways one could implement bivariance. You could
//! just do nothing at all, for example, or you could fully verify
//! that one of the two subtyping relationships hold. We choose to
//! thread a middle line: we relate types up to regions, but ignore
//! all region relationships.
//!
//! At one point, handling bivariance in this fashion was necessary
//! for inference, but I'm actually not sure if that is true anymore.
//! In particular, it might be enough to say (A,B) are bivariant for
//! all (A,B).

use middle::ty::{BuiltinBounds};
use middle::ty::{self, Ty};
use middle::ty::TyVar;
use middle::infer::combine::*;
use middle::infer::{cres};
use middle::infer::type_variable::{BiTo};
use util::ppaux::{Repr};

use syntax::ast::{Unsafety};

pub struct Bivariate<'f, 'tcx: 'f> {
    fields: CombineFields<'f, 'tcx>
}

#[allow(non_snake_case)]
pub fn Bivariate<'f, 'tcx>(cf: CombineFields<'f, 'tcx>) -> Bivariate<'f, 'tcx> {
    Bivariate { fields: cf }
}

impl<'f, 'tcx> Combine<'tcx> for Bivariate<'f, 'tcx> {
    fn tag(&self) -> String { "Bivariate".to_string() }
    fn fields<'a>(&'a self) -> &'a CombineFields<'a, 'tcx> { &self.fields }

    fn tys_with_variance(&self, v: ty::Variance, a: Ty<'tcx>, b: Ty<'tcx>)
                         -> cres<'tcx, Ty<'tcx>>
    {
        match v {
            ty::Invariant => self.equate().tys(a, b),
            ty::Covariant => self.tys(a, b),
            ty::Contravariant => self.tys(a, b),
            ty::Bivariant => self.tys(a, b),
        }
    }

    fn regions_with_variance(&self, v: ty::Variance, a: ty::Region, b: ty::Region)
                             -> cres<'tcx, ty::Region>
    {
        match v {
            ty::Invariant => self.equate().regions(a, b),
            ty::Covariant => self.regions(a, b),
            ty::Contravariant => self.regions(a, b),
            ty::Bivariant => self.regions(a, b),
        }
    }

    fn regions(&self, a: ty::Region, _: ty::Region) -> cres<'tcx, ty::Region> {
        Ok(a)
    }

    fn mts(&self, a: &ty::mt<'tcx>, b: &ty::mt<'tcx>) -> cres<'tcx, ty::mt<'tcx>> {
        debug!("mts({} <: {})",
               a.repr(self.fields.infcx.tcx),
               b.repr(self.fields.infcx.tcx));

        if a.mutbl != b.mutbl { return Err(ty::terr_mutability); }
        let t = try!(self.tys(a.ty, b.ty));
        Ok(ty::mt { mutbl: a.mutbl, ty: t })
    }

    fn unsafeties(&self, a: Unsafety, b: Unsafety) -> cres<'tcx, Unsafety> {
        if a != b {
            Err(ty::terr_unsafety_mismatch(expected_found(self, a, b)))
        } else {
            Ok(a)
        }
    }

    fn builtin_bounds(&self,
                      a: BuiltinBounds,
                      b: BuiltinBounds)
                      -> cres<'tcx, BuiltinBounds>
    {
        if a != b {
            Err(ty::terr_builtin_bounds(expected_found(self, a, b)))
        } else {
            Ok(a)
        }
    }

    fn tys(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> cres<'tcx, Ty<'tcx>> {
        debug!("{}.tys({}, {})", self.tag(),
               a.repr(self.fields.infcx.tcx), b.repr(self.fields.infcx.tcx));
        if a == b { return Ok(a); }

        let infcx = self.fields.infcx;
        let a = infcx.type_variables.borrow().replace_if_possible(a);
        let b = infcx.type_variables.borrow().replace_if_possible(b);
        match (&a.sty, &b.sty) {
            (&ty::ty_infer(TyVar(a_id)), &ty::ty_infer(TyVar(b_id))) => {
                infcx.type_variables.borrow_mut().relate_vars(a_id, BiTo, b_id);
                Ok(a)
            }

            (&ty::ty_infer(TyVar(a_id)), _) => {
                try!(self.fields.instantiate(b, BiTo, a_id));
                Ok(a)
            }

            (_, &ty::ty_infer(TyVar(b_id))) => {
                try!(self.fields.instantiate(a, BiTo, b_id));
                Ok(a)
            }

            _ => {
                super_tys(self, a, b)
            }
        }
    }

    fn binders<T>(&self, a: &ty::Binder<T>, b: &ty::Binder<T>) -> cres<'tcx, ty::Binder<T>>
        where T : Combineable<'tcx>
    {
        let a1 = ty::erase_late_bound_regions(self.tcx(), a);
        let b1 = ty::erase_late_bound_regions(self.tcx(), b);
        let c = try!(Combineable::combine(self, &a1, &b1));
        Ok(ty::Binder(c))
    }
}
