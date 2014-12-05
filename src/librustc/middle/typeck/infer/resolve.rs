// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{InferCtxt, fixup_err, fres, unresolved_ty, unresolved_int_ty, unresolved_float_ty};
use middle::ty::{mod, Ty};
use middle::ty_fold::{mod, TypeFoldable};
use util::ppaux::Repr;

///////////////////////////////////////////////////////////////////////////
// OPPORTUNISTIC TYPE RESOLVER

/// The opportunistic type resolver can be used at any time. It simply replaces
/// type variables that have been unified with the things they have
/// been unified with (similar to `shallow_resolve`, but deep). This is
/// useful for printing messages etc but also required at various
/// points for correctness.
pub struct OpportunisticTypeResolver<'a, 'tcx:'a> {
    infcx: &'a InferCtxt<'a, 'tcx>,
}

impl<'a, 'tcx> OpportunisticTypeResolver<'a, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'a, 'tcx>) -> OpportunisticTypeResolver<'a, 'tcx> {
        OpportunisticTypeResolver { infcx: infcx }
    }
}

impl<'a, 'tcx> ty_fold::TypeFolder<'tcx> for OpportunisticTypeResolver<'a, 'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !ty::type_has_ty_infer(t) {
            t // micro-optimize -- if there is nothing in this type that this fold affects...
        } else {
            let t0 = self.infcx.shallow_resolve(t);
            ty_fold::super_fold_ty(self, t0)
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// FULL TYPE RESOLUTION

/// Full type resolution replaces all type and region variables with
/// their concrete results. If any variable cannot be replaced (never unified, etc)
/// then an `Err` result is returned.
pub fn fully_resolve<'a, 'tcx, T>(infcx: &InferCtxt<'a,'tcx>, value: &T) -> fres<T>
    where T : TypeFoldable<'tcx>
{
    let mut full_resolver = FullTypeResolver { infcx: infcx, err: None };
    let result = value.fold_with(&mut full_resolver);
    match full_resolver.err {
        None => Ok(result),
        Some(e) => Err(e),
    }
}

// N.B. This type is not public because the protocol around checking the
// `err` field is not enforcable otherwise.
struct FullTypeResolver<'a, 'tcx:'a> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    err: Option<fixup_err>,
}

impl<'a, 'tcx> ty_fold::TypeFolder<'tcx> for FullTypeResolver<'a, 'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !ty::type_needs_infer(t) {
            t // micro-optimize -- if there is nothing in this type that this fold affects...
        } else {
            let t = self.infcx.shallow_resolve(t);
            match t.sty {
                ty::ty_infer(ty::TyVar(vid)) => {
                    self.err = Some(unresolved_ty(vid));
                    ty::mk_err()
                }
                ty::ty_infer(ty::IntVar(vid)) => {
                    self.err = Some(unresolved_int_ty(vid));
                    ty::mk_err()
                }
                ty::ty_infer(ty::FloatVar(vid)) => {
                    self.err = Some(unresolved_float_ty(vid));
                    ty::mk_err()
                }
                ty::ty_infer(_) => {
                    self.infcx.tcx.sess.bug(
                        format!("Unexpected type in full type resolver: {}",
                                t.repr(self.infcx.tcx))[]);
                }
                _ => {
                    ty_fold::super_fold_ty(self, t)
                }
            }
        }
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        match r {
          ty::ReInfer(ty::ReVar(rid)) => self.infcx.region_vars.resolve_var(rid),
          _ => r,
        }
    }
}

