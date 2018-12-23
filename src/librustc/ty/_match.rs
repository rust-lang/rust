// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ty::{self, Ty, TyCtxt};
use ty::error::TypeError;
use ty::relate::{self, Relate, TypeRelation, RelateResult};

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
pub struct Match<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    trait_object_mode: relate::TraitObjectMode
}

impl<'a, 'gcx, 'tcx> Match<'a, 'gcx, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>,
               trait_object_mode: relate::TraitObjectMode)
               -> Match<'a, 'gcx, 'tcx> {
        Match { tcx, trait_object_mode }
    }
}

impl<'a, 'gcx, 'tcx> TypeRelation<'a, 'gcx, 'tcx> for Match<'a, 'gcx, 'tcx> {
    fn tag(&self) -> &'static str { "Match" }
    fn trait_object_mode(&self) -> relate::TraitObjectMode {
        self.trait_object_mode
    }

    fn tcx(&self) -> TyCtxt<'a, 'gcx, 'tcx> { self.tcx }
    fn a_is_expected(&self) -> bool { true } // irrelevant

    fn relate_with_variance<T: Relate<'tcx>>(&mut self,
                                             _: ty::Variance,
                                             a: &T,
                                             b: &T)
                                             -> RelateResult<'tcx, T>
    {
        self.relate(a, b)
    }

    fn regions(&mut self, a: ty::Region<'tcx>, b: ty::Region<'tcx>)
               -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug!("{}.regions({:?}, {:?})",
               self.tag(),
               a,
               b);
        Ok(a)
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        debug!("{}.tys({:?}, {:?})", self.tag(),
               a, b);
        if a == b { return Ok(a); }

        match (&a.sty, &b.sty) {
            (_, &ty::Infer(ty::FreshTy(_))) |
            (_, &ty::Infer(ty::FreshIntTy(_))) |
            (_, &ty::Infer(ty::FreshFloatTy(_))) => {
                Ok(a)
            }

            (&ty::Infer(_), _) |
            (_, &ty::Infer(_)) => {
                Err(TypeError::Sorts(relate::expected_found(self, &a, &b)))
            }

            (&ty::Error, _) | (_, &ty::Error) => {
                Ok(self.tcx().types.err)
            }

            _ => {
                relate::super_relate_tys(self, a, b)
            }
        }
    }

    fn binders<T>(&mut self, a: &ty::Binder<T>, b: &ty::Binder<T>)
                  -> RelateResult<'tcx, ty::Binder<T>>
        where T: Relate<'tcx>
    {
        Ok(ty::Binder::bind(self.relate(a.skip_binder(), b.skip_binder())?))
    }
}
