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
use middle::ty;
use middle::ty::TyVar;
use middle::typeck::infer::combine::*;
use middle::typeck::infer::{cres};
use middle::typeck::infer::glb::Glb;
use middle::typeck::infer::InferCtxt;
use middle::typeck::infer::lub::Lub;
use middle::typeck::infer::sub::Sub;
use middle::typeck::infer::{TypeTrace, Subtype};
use middle::typeck::infer::type_variable::{EqTo};
use util::ppaux::{Repr};

use syntax::ast::{Onceness, FnStyle};

pub struct Equate<'f> {
    fields: CombineFields<'f>
}

#[allow(non_snake_case)]
pub fn Equate<'f>(cf: CombineFields<'f>) -> Equate<'f> {
    Equate { fields: cf }
}

impl<'f> Combine for Equate<'f> {
    fn infcx<'a>(&'a self) -> &'a InferCtxt<'a> { self.fields.infcx }
    fn tag(&self) -> String { "eq".to_string() }
    fn a_is_expected(&self) -> bool { self.fields.a_is_expected }
    fn trace(&self) -> TypeTrace { self.fields.trace.clone() }

    fn equate<'a>(&'a self) -> Equate<'a> { Equate(self.fields.clone()) }
    fn sub<'a>(&'a self) -> Sub<'a> { Sub(self.fields.clone()) }
    fn lub<'a>(&'a self) -> Lub<'a> { Lub(self.fields.clone()) }
    fn glb<'a>(&'a self) -> Glb<'a> { Glb(self.fields.clone()) }

    fn contratys(&self, a: ty::t, b: ty::t) -> cres<ty::t> {
        self.tys(a, b)
    }

    fn contraregions(&self, a: ty::Region, b: ty::Region) -> cres<ty::Region> {
        self.regions(a, b)
    }

    fn regions(&self, a: ty::Region, b: ty::Region) -> cres<ty::Region> {
        debug!("{}.regions({}, {})",
               self.tag(),
               a.repr(self.fields.infcx.tcx),
               b.repr(self.fields.infcx.tcx));
        self.infcx().region_vars.make_eqregion(Subtype(self.trace()), a, b);
        Ok(a)
    }

    fn mts(&self, a: &ty::mt, b: &ty::mt) -> cres<ty::mt> {
        debug!("mts({} <: {})",
               a.repr(self.fields.infcx.tcx),
               b.repr(self.fields.infcx.tcx));

        if a.mutbl != b.mutbl { return Err(ty::terr_mutability); }
        let t = try!(self.tys(a.ty, b.ty));
        Ok(ty::mt { mutbl: a.mutbl, ty: t })
    }

    fn fn_styles(&self, a: FnStyle, b: FnStyle) -> cres<FnStyle> {
        if a != b {
            Err(ty::terr_fn_style_mismatch(expected_found(self, a, b)))
        } else {
            Ok(a)
        }
    }

    fn oncenesses(&self, a: Onceness, b: Onceness) -> cres<Onceness> {
        if a != b {
            Err(ty::terr_onceness_mismatch(expected_found(self, a, b)))
        } else {
            Ok(a)
        }
    }

    fn builtin_bounds(&self,
                      a: BuiltinBounds,
                      b: BuiltinBounds)
                      -> cres<BuiltinBounds>
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

    fn tys(&self, a: ty::t, b: ty::t) -> cres<ty::t> {
        debug!("{}.tys({}, {})", self.tag(),
               a.repr(self.fields.infcx.tcx), b.repr(self.fields.infcx.tcx));
        if a == b { return Ok(a); }

        let infcx = self.fields.infcx;
        let a = infcx.type_variables.borrow().replace_if_possible(a);
        let b = infcx.type_variables.borrow().replace_if_possible(b);
        match (&ty::get(a).sty, &ty::get(b).sty) {
            (&ty::ty_bot, &ty::ty_bot) => {
                Ok(a)
            }

            (&ty::ty_bot, _) |
            (_, &ty::ty_bot) => {
                Err(ty::terr_sorts(expected_found(self, a, b)))
            }

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

    fn fn_sigs(&self, a: &ty::FnSig, b: &ty::FnSig) -> cres<ty::FnSig> {
        try!(self.sub().fn_sigs(a, b));
        self.sub().fn_sigs(b, a)
    }
}
