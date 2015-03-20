// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::ty::{self, Ty};
use middle::ty_relate::{self, Relate, TypeRelation, RelateResult};
use util::ppaux::Repr;

/// A type "A" *matches* "B" if the fresh types in B could be
/// substituted with values so as to make it equal to A. Matching is
/// intended to be used only on freshened types, and it basically
/// indicates if the non-freshened versions of A and B could have been
/// unified.
///
/// It is only an approximation. If it yields false, unification would
/// definitely fail, but a true result doesn't mean unification would
/// succeed. This is because we don't track the "side-constraints" on
/// type variables, nor do we track if the same freshened type appears
/// more than once. To some extent these approximations could be
/// fixed, given effort.
///
/// Like subtyping, matching is really a binary relation, so the only
/// important thing about the result is Ok/Err. Also, matching never
/// affects any type variables or unification state.
pub struct Match<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>
}

impl<'a, 'tcx> Match<'a, 'tcx> {
    pub fn new(tcx: &'a ty::ctxt<'tcx>) -> Match<'a, 'tcx> {
        Match { tcx: tcx }
    }
}

impl<'a, 'tcx> TypeRelation<'a, 'tcx> for Match<'a, 'tcx> {
    fn tag(&self) -> &'static str { "Match" }
    fn tcx(&self) -> &'a ty::ctxt<'tcx> { self.tcx }
    fn a_is_expected(&self) -> bool { true } // irrelevant

    fn relate_with_variance<T:Relate<'a,'tcx>>(&mut self,
                                               _: ty::Variance,
                                               a: &T,
                                               b: &T)
                                               -> RelateResult<'tcx, T>
    {
        self.relate(a, b)
    }

    fn regions(&mut self, a: ty::Region, b: ty::Region) -> RelateResult<'tcx, ty::Region> {
        debug!("{}.regions({}, {})",
               self.tag(),
               a.repr(self.tcx()),
               b.repr(self.tcx()));
        Ok(a)
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        debug!("{}.tys({}, {})", self.tag(),
               a.repr(self.tcx()), b.repr(self.tcx()));
        if a == b { return Ok(a); }

        match (&a.sty, &b.sty) {
            (_, &ty::ty_infer(ty::FreshTy(_))) |
            (_, &ty::ty_infer(ty::FreshIntTy(_))) => {
                Ok(a)
            }

            (&ty::ty_infer(_), _) |
            (_, &ty::ty_infer(_)) => {
                Err(ty::terr_sorts(ty_relate::expected_found(self, &a, &b)))
            }

            (&ty::ty_err, _) | (_, &ty::ty_err) => {
                Ok(self.tcx().types.err)
            }

            _ => {
                ty_relate::super_relate_tys(self, a, b)
            }
        }
    }

    fn binders<T>(&mut self, a: &ty::Binder<T>, b: &ty::Binder<T>)
                  -> RelateResult<'tcx, ty::Binder<T>>
        where T: Relate<'a,'tcx>
    {
        Ok(ty::Binder(try!(self.relate(a.skip_binder(), b.skip_binder()))))
    }
}
