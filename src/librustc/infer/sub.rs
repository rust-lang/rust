// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::SubregionOrigin;
use super::combine::{CombineFields, RelationDir};

use traits::Obligation;
use ty::{self, Ty, TyCtxt};
use ty::TyVar;
use ty::fold::TypeFoldable;
use ty::relate::{self, Cause, Relate, RelateResult, TypeRelation};
use std::mem;

/// Ensures `a` is made a subtype of `b`. Returns `a` on success.
pub struct Sub<'combine, 'infcx: 'combine, 'gcx: 'infcx+'tcx, 'tcx: 'infcx> {
    fields: &'combine mut CombineFields<'infcx, 'gcx, 'tcx>,
    a_is_expected: bool,
}

impl<'combine, 'infcx, 'gcx, 'tcx> Sub<'combine, 'infcx, 'gcx, 'tcx> {
    pub fn new(f: &'combine mut CombineFields<'infcx, 'gcx, 'tcx>, a_is_expected: bool)
        -> Sub<'combine, 'infcx, 'gcx, 'tcx>
    {
        Sub { fields: f, a_is_expected: a_is_expected }
    }

    fn with_expected_switched<R, F: FnOnce(&mut Self) -> R>(&mut self, f: F) -> R {
        self.a_is_expected = !self.a_is_expected;
        let result = f(self);
        self.a_is_expected = !self.a_is_expected;
        result
    }
}

impl<'combine, 'infcx, 'gcx, 'tcx> TypeRelation<'infcx, 'gcx, 'tcx>
    for Sub<'combine, 'infcx, 'gcx, 'tcx>
{
    fn tag(&self) -> &'static str { "Sub" }
    fn trait_object_mode(&self) -> relate::TraitObjectMode {
        self.fields.infcx.trait_object_mode()
    }

    fn tcx(&self) -> TyCtxt<'infcx, 'gcx, 'tcx> { self.fields.infcx.tcx }
    fn a_is_expected(&self) -> bool { self.a_is_expected }

    fn with_cause<F,R>(&mut self, cause: Cause, f: F) -> R
        where F: FnOnce(&mut Self) -> R
    {
        debug!("sub with_cause={:?}", cause);
        let old_cause = mem::replace(&mut self.fields.cause, Some(cause));
        let r = f(self);
        debug!("sub old_cause={:?}", old_cause);
        self.fields.cause = old_cause;
        r
    }

    fn relate_with_variance<T: Relate<'tcx>>(&mut self,
                                             variance: ty::Variance,
                                             a: &T,
                                             b: &T)
                                             -> RelateResult<'tcx, T>
    {
        match variance {
            ty::Invariant => self.fields.equate(self.a_is_expected).relate(a, b),
            ty::Covariant => self.relate(a, b),
            ty::Bivariant => Ok(a.clone()),
            ty::Contravariant => self.with_expected_switched(|this| { this.relate(b, a) }),
        }
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        debug!("{}.tys({:?}, {:?})", self.tag(), a, b);

        if a == b { return Ok(a); }

        let infcx = self.fields.infcx;
        let a = infcx.type_variables.borrow_mut().replace_if_possible(a);
        let b = infcx.type_variables.borrow_mut().replace_if_possible(b);
        match (&a.sty, &b.sty) {
            (&ty::Infer(TyVar(a_vid)), &ty::Infer(TyVar(b_vid))) => {
                // Shouldn't have any LBR here, so we can safely put
                // this under a binder below without fear of accidental
                // capture.
                assert!(!a.has_escaping_bound_vars());
                assert!(!b.has_escaping_bound_vars());

                // can't make progress on `A <: B` if both A and B are
                // type variables, so record an obligation. We also
                // have to record in the `type_variables` tracker that
                // the two variables are equal modulo subtyping, which
                // is important to the occurs check later on.
                infcx.type_variables.borrow_mut().sub(a_vid, b_vid);
                self.fields.obligations.push(
                    Obligation::new(
                        self.fields.trace.cause.clone(),
                        self.fields.param_env,
                        ty::Predicate::Subtype(
                            ty::Binder::dummy(ty::SubtypePredicate {
                                a_is_expected: self.a_is_expected,
                                a,
                                b,
                            }))));

                Ok(a)
            }
            (&ty::Infer(TyVar(a_id)), _) => {
                self.fields
                    .instantiate(b, RelationDir::SupertypeOf, a_id, !self.a_is_expected)?;
                Ok(a)
            }
            (_, &ty::Infer(TyVar(b_id))) => {
                self.fields.instantiate(a, RelationDir::SubtypeOf, b_id, self.a_is_expected)?;
                Ok(a)
            }

            (&ty::Error, _) | (_, &ty::Error) => {
                infcx.set_tainted_by_errors();
                Ok(self.tcx().types.err)
            }

            _ => {
                self.fields.infcx.super_combine_tys(self, a, b)?;
                Ok(a)
            }
        }
    }

    fn regions(&mut self, a: ty::Region<'tcx>, b: ty::Region<'tcx>)
               -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug!("{}.regions({:?}, {:?}) self.cause={:?}",
               self.tag(), a, b, self.fields.cause);

        // FIXME -- we have more fine-grained information available
        // from the "cause" field, we could perhaps give more tailored
        // error messages.
        let origin = SubregionOrigin::Subtype(self.fields.trace.clone());
        self.fields.infcx.borrow_region_constraints()
                         .make_subregion(origin, a, b);

        Ok(a)
    }

    fn binders<T>(&mut self, a: &ty::Binder<T>, b: &ty::Binder<T>)
                  -> RelateResult<'tcx, ty::Binder<T>>
        where T: Relate<'tcx>
    {
        self.fields.higher_ranked_sub(a, b, self.a_is_expected)
    }
}
