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
use super::{cres, CresCompare};
use super::equate::Equate;
use super::glb::Glb;
use super::higher_ranked::HigherRankedRelations;
use super::InferCtxt;
use super::lub::Lub;
use super::{TypeTrace, Subtype};
use super::type_variable::{SubtypeOf, SupertypeOf};

use middle::ty::{BuiltinBounds};
use middle::ty::{mod, Ty};
use middle::ty::TyVar;
use util::ppaux::{Repr};

use syntax::ast::{Onceness, FnStyle, MutImmutable, MutMutable};


/// "Greatest lower bound" (common subtype)
pub struct Sub<'f, 'tcx: 'f> {
    fields: CombineFields<'f, 'tcx>
}

#[allow(non_snake_case)]
pub fn Sub<'f, 'tcx>(cf: CombineFields<'f, 'tcx>) -> Sub<'f, 'tcx> {
    Sub { fields: cf }
}

impl<'f, 'tcx> Combine<'tcx> for Sub<'f, 'tcx> {
    fn infcx<'a>(&'a self) -> &'a InferCtxt<'a, 'tcx> { self.fields.infcx }
    fn tag(&self) -> String { "sub".to_string() }
    fn a_is_expected(&self) -> bool { self.fields.a_is_expected }
    fn trace(&self) -> TypeTrace<'tcx> { self.fields.trace.clone() }

    fn equate<'a>(&'a self) -> Equate<'a, 'tcx> { Equate(self.fields.clone()) }
    fn sub<'a>(&'a self) -> Sub<'a, 'tcx> { Sub(self.fields.clone()) }
    fn lub<'a>(&'a self) -> Lub<'a, 'tcx> { Lub(self.fields.clone()) }
    fn glb<'a>(&'a self) -> Glb<'a, 'tcx> { Glb(self.fields.clone()) }

    fn contratys(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> cres<'tcx, Ty<'tcx>> {
        Sub(self.fields.switch_expected()).tys(b, a)
    }

    fn contraregions(&self, a: ty::Region, b: ty::Region)
                     -> cres<'tcx, ty::Region> {
                         let opp = CombineFields {
                             a_is_expected: !self.fields.a_is_expected,
                             ..self.fields.clone()
                         };
                         Sub(opp).regions(b, a)
                     }

    fn regions(&self, a: ty::Region, b: ty::Region) -> cres<'tcx, ty::Region> {
        debug!("{}.regions({}, {})",
               self.tag(),
               a.repr(self.tcx()),
               b.repr(self.tcx()));
        self.infcx().region_vars.make_subregion(Subtype(self.trace()), a, b);
        Ok(a)
    }

    fn mts(&self, a: &ty::mt<'tcx>, b: &ty::mt<'tcx>) -> cres<'tcx, ty::mt<'tcx>> {
        debug!("mts({} <: {})",
               a.repr(self.tcx()),
               b.repr(self.tcx()));

        if a.mutbl != b.mutbl {
            return Err(ty::terr_mutability);
        }

        match b.mutbl {
            MutMutable => {
                // If supertype is mut, subtype must match exactly
                // (i.e., invariant if mut):
                try!(self.equate().tys(a.ty, b.ty));
            }
            MutImmutable => {
                // Otherwise we can be covariant:
                try!(self.tys(a.ty, b.ty));
            }
        }

        Ok(*a) // return is meaningless in sub, just return *a
    }

    fn fn_styles(&self, a: FnStyle, b: FnStyle) -> cres<'tcx, FnStyle> {
        self.lub().fn_styles(a, b).compare(b, || {
            ty::terr_fn_style_mismatch(expected_found(self, a, b))
        })
    }

    fn oncenesses(&self, a: Onceness, b: Onceness) -> cres<'tcx, Onceness> {
        self.lub().oncenesses(a, b).compare(b, || {
            ty::terr_onceness_mismatch(expected_found(self, a, b))
        })
    }

    fn builtin_bounds(&self, a: BuiltinBounds, b: BuiltinBounds)
                      -> cres<'tcx, BuiltinBounds> {
        // More bounds is a subtype of fewer bounds.
        //
        // e.g., fn:Copy() <: fn(), because the former is a function
        // that only closes over copyable things, but the latter is
        // any function at all.
        if a.is_superset(&b) {
            Ok(a)
        } else {
            Err(ty::terr_builtin_bounds(expected_found(self, a, b)))
        }
    }

    fn tys(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> cres<'tcx, Ty<'tcx>> {
        debug!("{}.tys({}, {})", self.tag(),
               a.repr(self.tcx()), b.repr(self.tcx()));
        if a == b { return Ok(a); }

        let infcx = self.fields.infcx;
        let a = infcx.type_variables.borrow().replace_if_possible(a);
        let b = infcx.type_variables.borrow().replace_if_possible(b);
        match (&a.sty, &b.sty) {
            (&ty::ty_infer(TyVar(a_id)), &ty::ty_infer(TyVar(b_id))) => {
                infcx.type_variables
                    .borrow_mut()
                    .relate_vars(a_id, SubtypeOf, b_id);
                Ok(a)
            }
            (&ty::ty_infer(TyVar(a_id)), _) => {
                try!(self.fields
                       .switch_expected()
                       .instantiate(b, SupertypeOf, a_id));
                Ok(a)
            }
            (_, &ty::ty_infer(TyVar(b_id))) => {
                try!(self.fields.instantiate(a, SubtypeOf, b_id));
                Ok(a)
            }

            (&ty::ty_err, _) | (_, &ty::ty_err) => {
                Ok(ty::mk_err())
            }

            _ => {
                super_tys(self, a, b)
            }
        }
    }

    fn fn_sigs(&self, a: &ty::FnSig<'tcx>, b: &ty::FnSig<'tcx>)
               -> cres<'tcx, ty::FnSig<'tcx>> {
        self.higher_ranked_sub(a, b)
    }

    fn trait_refs(&self, a: &ty::TraitRef<'tcx>, b: &ty::TraitRef<'tcx>)
                  -> cres<'tcx, ty::TraitRef<'tcx>> {
        self.higher_ranked_sub(a, b)
    }
}

