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

use super::combine::{self, CombineFields};
use super::type_variable::{BiTo};

use middle::ty::{self, Ty};
use middle::ty::TyVar;
use middle::ty_relate::{Relate, RelateResult, TypeRelation};
use util::ppaux::{Repr};

pub struct Bivariate<'a, 'tcx: 'a> {
    fields: CombineFields<'a, 'tcx>
}

impl<'a, 'tcx> Bivariate<'a, 'tcx> {
    pub fn new(fields: CombineFields<'a, 'tcx>) -> Bivariate<'a, 'tcx> {
        Bivariate { fields: fields }
    }
}

impl<'a, 'tcx> TypeRelation<'a, 'tcx> for Bivariate<'a, 'tcx> {
    fn tag(&self) -> &'static str { "Bivariate" }

    fn tcx(&self) -> &'a ty::ctxt<'tcx> { self.fields.tcx() }

    fn a_is_expected(&self) -> bool { self.fields.a_is_expected }

    fn relate_with_variance<T:Relate<'a,'tcx>>(&mut self,
                                               variance: ty::Variance,
                                               a: &T,
                                               b: &T)
                                               -> RelateResult<'tcx, T>
    {
        match variance {
            // If we have Foo<A> and Foo is invariant w/r/t A,
            // and we want to assert that
            //
            //     Foo<A> <: Foo<B> ||
            //     Foo<B> <: Foo<A>
            //
            // then still A must equal B.
            ty::Invariant => self.relate(a, b),

            ty::Covariant => self.relate(a, b),
            ty::Bivariant => self.relate(a, b),
            ty::Contravariant => self.relate(a, b),
        }
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
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
                combine::super_combine_tys(self.fields.infcx, self, a, b)
            }
        }
    }

    fn regions(&mut self, a: ty::Region, _: ty::Region) -> RelateResult<'tcx, ty::Region> {
        Ok(a)
    }

    fn binders<T>(&mut self, a: &ty::Binder<T>, b: &ty::Binder<T>)
                  -> RelateResult<'tcx, ty::Binder<T>>
        where T: Relate<'a,'tcx>
    {
        let a1 = ty::erase_late_bound_regions(self.tcx(), a);
        let b1 = ty::erase_late_bound_regions(self.tcx(), b);
        let c = try!(self.relate(&a1, &b1));
        Ok(ty::Binder(c))
    }
}
