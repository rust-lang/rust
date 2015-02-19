// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::ty::{BuiltinBounds};
use middle::ty::{self, Ty};
use middle::ty::TyVar;
use middle::infer::combine::*;
use middle::infer::{cres};
use middle::infer::{Subtype};
use middle::infer::type_variable::{EqTo};
use util::ppaux::{Repr};

use syntax::ast::Unsafety;

pub struct Equate<'f, 'tcx: 'f> {
    fields: CombineFields<'f, 'tcx>
}

#[allow(non_snake_case)]
pub fn Equate<'f, 'tcx>(cf: CombineFields<'f, 'tcx>) -> Equate<'f, 'tcx> {
    Equate { fields: cf }
}

impl<'f, 'tcx> Combine<'tcx> for Equate<'f, 'tcx> {
    fn tag(&self) -> String { "Equate".to_string() }
    fn fields<'a>(&'a self) -> &'a CombineFields<'a, 'tcx> { &self.fields }

    fn tys_with_variance(&self, _: ty::Variance, a: Ty<'tcx>, b: Ty<'tcx>)
                         -> cres<'tcx, Ty<'tcx>>
    {
        // Once we're equating, it doesn't matter what the variance is.
        self.tys(a, b)
    }

    fn regions_with_variance(&self, _: ty::Variance, a: ty::Region, b: ty::Region)
                             -> cres<'tcx, ty::Region>
    {
        // Once we're equating, it doesn't matter what the variance is.
        self.regions(a, b)
    }

    fn regions(&self, a: ty::Region, b: ty::Region) -> cres<'tcx, ty::Region> {
        debug!("{}.regions({}, {})",
               self.tag(),
               a.repr(self.fields.infcx.tcx),
               b.repr(self.fields.infcx.tcx));
        self.infcx().region_vars.make_eqregion(Subtype(self.trace()), a, b);
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
        // More bounds is a subtype of fewer bounds.
        //
        // e.g., fn:Copy() <: fn(), because the former is a function
        // that only closes over copyable things, but the latter is
        // any function at all.
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
                infcx.type_variables.borrow_mut().relate_vars(a_id, EqTo, b_id);
                Ok(a)
            }

            (&ty::ty_infer(TyVar(a_id)), _) => {
                try!(self.fields.instantiate(b, EqTo, a_id));
                Ok(a)
            }

            (_, &ty::ty_infer(TyVar(b_id))) => {
                try!(self.fields.instantiate(a, EqTo, b_id));
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
        try!(self.sub().binders(a, b));
        self.sub().binders(b, a)
    }
}
