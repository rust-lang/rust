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
use super::higher_ranked::HigherRankedRelations;
use super::{Subtype};
use super::type_variable::{SubtypeOf, SupertypeOf};

use middle::ty::{BuiltinBounds};
use middle::ty::{self, Ty};
use middle::ty::TyVar;
use util::ppaux::{Repr};

use syntax::ast::{MutImmutable, MutMutable, Unsafety};


/// "Greatest lower bound" (common subtype)
pub struct Sub<'f, 'tcx: 'f> {
    fields: CombineFields<'f, 'tcx>
}

#[allow(non_snake_case)]
pub fn Sub<'f, 'tcx>(cf: CombineFields<'f, 'tcx>) -> Sub<'f, 'tcx> {
    Sub { fields: cf }
}

impl<'f, 'tcx> Combine<'tcx> for Sub<'f, 'tcx> {
    fn tag(&self) -> String { "Sub".to_string() }
    fn fields<'a>(&'a self) -> &'a CombineFields<'a, 'tcx> { &self.fields }

    fn tys_with_variance(&self, v: ty::Variance, a: Ty<'tcx>, b: Ty<'tcx>)
                         -> cres<'tcx, Ty<'tcx>>
    {
        match v {
            ty::Invariant => self.equate().tys(a, b),
            ty::Covariant => self.tys(a, b),
            ty::Bivariant => self.bivariate().tys(a, b),
            ty::Contravariant => Sub(self.fields.switch_expected()).tys(b, a),
        }
    }

    fn regions_with_variance(&self, v: ty::Variance, a: ty::Region, b: ty::Region)
                             -> cres<'tcx, ty::Region>
    {
        match v {
            ty::Invariant => self.equate().regions(a, b),
            ty::Covariant => self.regions(a, b),
            ty::Bivariant => self.bivariate().regions(a, b),
            ty::Contravariant => Sub(self.fields.switch_expected()).regions(b, a),
        }
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

    fn unsafeties(&self, a: Unsafety, b: Unsafety) -> cres<'tcx, Unsafety> {
        self.lub().unsafeties(a, b).compare(b, || {
            ty::terr_unsafety_mismatch(expected_found(self, a, b))
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
                Ok(self.tcx().types.err)
            }

            _ => {
                super_tys(self, a, b)
            }
        }
    }

    fn binders<T>(&self, a: &ty::Binder<T>, b: &ty::Binder<T>) -> cres<'tcx, ty::Binder<T>>
        where T : Combineable<'tcx>
    {
        self.higher_ranked_sub(a, b)
    }
}

