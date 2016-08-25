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

use super::combine::CombineFields;
use super::type_variable::{BiTo};

use ty::{self, Ty, TyCtxt};
use ty::TyVar;
use ty::relate::{Relate, RelateResult, TypeRelation};

pub struct Bivariate<'combine, 'infcx: 'combine, 'gcx: 'infcx+'tcx, 'tcx: 'infcx> {
    fields: &'combine mut CombineFields<'infcx, 'gcx, 'tcx>,
    a_is_expected: bool,
}

impl<'combine, 'infcx, 'gcx, 'tcx> Bivariate<'combine, 'infcx, 'gcx, 'tcx> {
    pub fn new(fields: &'combine mut CombineFields<'infcx, 'gcx, 'tcx>, a_is_expected: bool)
        -> Bivariate<'combine, 'infcx, 'gcx, 'tcx>
    {
        Bivariate { fields: fields, a_is_expected: a_is_expected }
    }
}

impl<'combine, 'infcx, 'gcx, 'tcx> TypeRelation<'infcx, 'gcx, 'tcx>
    for Bivariate<'combine, 'infcx, 'gcx, 'tcx>
{
    fn tag(&self) -> &'static str { "Bivariate" }

    fn tcx(&self) -> TyCtxt<'infcx, 'gcx, 'tcx> { self.fields.tcx() }

    fn a_is_expected(&self) -> bool { self.a_is_expected }

    fn relate_with_variance<T: Relate<'tcx>>(&mut self,
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
        debug!("{}.tys({:?}, {:?})", self.tag(),
               a, b);
        if a == b { return Ok(a); }

        let infcx = self.fields.infcx;
        let a = infcx.type_variables.borrow_mut().replace_if_possible(a);
        let b = infcx.type_variables.borrow_mut().replace_if_possible(b);
        match (&a.sty, &b.sty) {
            (&ty::TyInfer(TyVar(a_id)), &ty::TyInfer(TyVar(b_id))) => {
                infcx.type_variables.borrow_mut().relate_vars(a_id, BiTo, b_id);
                Ok(a)
            }

            (&ty::TyInfer(TyVar(a_id)), _) => {
                self.fields.instantiate(b, BiTo, a_id, self.a_is_expected)?;
                Ok(a)
            }

            (_, &ty::TyInfer(TyVar(b_id))) => {
                self.fields.instantiate(a, BiTo, b_id, self.a_is_expected)?;
                Ok(a)
            }

            _ => {
                self.fields.infcx.super_combine_tys(self, a, b)
            }
        }
    }

    fn regions(&mut self, a: &'tcx ty::Region, _: &'tcx ty::Region)
               -> RelateResult<'tcx, &'tcx ty::Region> {
        Ok(a)
    }

    fn binders<T>(&mut self, a: &ty::Binder<T>, b: &ty::Binder<T>)
                  -> RelateResult<'tcx, ty::Binder<T>>
        where T: Relate<'tcx>
    {
        let a1 = self.tcx().erase_late_bound_regions(a);
        let b1 = self.tcx().erase_late_bound_regions(b);
        let c = self.relate(&a1, &b1)?;
        Ok(ty::Binder(c))
    }
}
