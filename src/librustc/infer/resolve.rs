// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{InferCtxt, FixupError, FixupResult};
use ty::{self, Ty, TyCtxt, TypeFoldable};
use ty::fold::TypeFolder;

///////////////////////////////////////////////////////////////////////////
// OPPORTUNISTIC TYPE RESOLVER

/// The opportunistic type resolver can be used at any time. It simply replaces
/// type variables that have been unified with the things they have
/// been unified with (similar to `shallow_resolve`, but deep). This is
/// useful for printing messages etc but also required at various
/// points for correctness.
pub struct OpportunisticTypeResolver<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> OpportunisticTypeResolver<'a, 'gcx, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>) -> Self {
        OpportunisticTypeResolver { infcx: infcx }
    }
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for OpportunisticTypeResolver<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.has_infer_types() {
            t // micro-optimize -- if there is nothing in this type that this fold affects...
        } else {
            let t0 = self.infcx.shallow_resolve(t);
            t0.super_fold_with(self)
        }
    }
}

/// The opportunistic type and region resolver is similar to the
/// opportunistic type resolver, but also opportunistly resolves
/// regions. It is useful for canonicalization.
pub struct OpportunisticTypeAndRegionResolver<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> OpportunisticTypeAndRegionResolver<'a, 'gcx, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>) -> Self {
        OpportunisticTypeAndRegionResolver { infcx: infcx }
    }
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for OpportunisticTypeAndRegionResolver<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.needs_infer() {
            t // micro-optimize -- if there is nothing in this type that this fold affects...
        } else {
            let t0 = self.infcx.shallow_resolve(t);
            t0.super_fold_with(self)
        }
    }

    fn fold_region(&mut self, r: &'tcx ty::Region) -> &'tcx ty::Region {
        match *r {
            ty::ReVar(rid) => self.infcx.region_vars.opportunistic_resolve_var(rid),
            _ => r,
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// FULL TYPE RESOLUTION

/// Full type resolution replaces all type and region variables with
/// their concrete results. If any variable cannot be replaced (never unified, etc)
/// then an `Err` result is returned.
pub fn fully_resolve<'a, 'gcx, 'tcx, T>(infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                                        value: &T) -> FixupResult<T>
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
struct FullTypeResolver<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    err: Option<FixupError>,
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for FullTypeResolver<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.needs_infer() {
            t // micro-optimize -- if there is nothing in this type that this fold affects...
        } else {
            let t = self.infcx.shallow_resolve(t);
            match t.sty {
                ty::TyInfer(ty::TyVar(vid)) => {
                    self.err = Some(FixupError::UnresolvedTy(vid));
                    self.tcx().types.err
                }
                ty::TyInfer(ty::IntVar(vid)) => {
                    self.err = Some(FixupError::UnresolvedIntTy(vid));
                    self.tcx().types.err
                }
                ty::TyInfer(ty::FloatVar(vid)) => {
                    self.err = Some(FixupError::UnresolvedFloatTy(vid));
                    self.tcx().types.err
                }
                ty::TyInfer(_) => {
                    bug!("Unexpected type in full type resolver: {:?}", t);
                }
                _ => {
                    t.super_fold_with(self)
                }
            }
        }
    }

    fn fold_region(&mut self, r: &'tcx ty::Region) -> &'tcx ty::Region {
        match *r {
            ty::ReVar(rid) => self.infcx.region_vars.resolve_var(rid),
            _ => r,
        }
    }
}
