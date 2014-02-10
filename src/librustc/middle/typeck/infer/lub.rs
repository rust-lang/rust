// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use middle::ty::{BuiltinBounds};
use middle::ty::RegionVid;
use middle::ty;
use middle::typeck::infer::then;
use middle::typeck::infer::combine::*;
use middle::typeck::infer::glb::Glb;
use middle::typeck::infer::lattice::*;
use middle::typeck::infer::sub::Sub;
use middle::typeck::infer::to_str::InferStr;
use middle::typeck::infer::{cres, InferCtxt};
use middle::typeck::infer::fold_regions_in_sig;
use middle::typeck::infer::{TypeTrace, Subtype};
use std::hashmap::HashMap;
use syntax::ast::{Many, Once, NodeId};
use syntax::ast::{ExternFn, ImpureFn, UnsafeFn};
use syntax::ast::{Onceness, Purity};
use util::ppaux::mt_to_str;

pub struct Lub<'f>(CombineFields<'f>);  // least-upper-bound: common supertype

impl<'f> Lub<'f> {
    pub fn get_ref<'a>(&'a self) -> &'a CombineFields<'f> { let Lub(ref v) = *self; v }
    pub fn bot_ty(&self, b: ty::t) -> cres<ty::t> { Ok(b) }
    pub fn ty_bot(&self, b: ty::t) -> cres<ty::t> {
        self.bot_ty(b) // commutative
    }
}

impl<'f> Combine for Lub<'f> {
    fn infcx<'a>(&'a self) -> &'a InferCtxt { self.get_ref().infcx }
    fn tag(&self) -> ~str { ~"lub" }
    fn a_is_expected(&self) -> bool { self.get_ref().a_is_expected }
    fn trace(&self) -> TypeTrace { self.get_ref().trace }

    fn sub<'a>(&'a self) -> Sub<'a> { Sub(*self.get_ref()) }
    fn lub<'a>(&'a self) -> Lub<'a> { Lub(*self.get_ref()) }
    fn glb<'a>(&'a self) -> Glb<'a> { Glb(*self.get_ref()) }

    fn mts(&self, a: &ty::mt, b: &ty::mt) -> cres<ty::mt> {
        let tcx = self.get_ref().infcx.tcx;

        debug!("{}.mts({}, {})",
               self.tag(),
               mt_to_str(tcx, a),
               mt_to_str(tcx, b));

        if a.mutbl != b.mutbl {
            return Err(ty::terr_mutability)
        }

        let m = a.mutbl;
        match m {
          MutImmutable => {
            self.tys(a.ty, b.ty).and_then(|t| Ok(ty::mt {ty: t, mutbl: m}) )
          }

          MutMutable => {
            self.get_ref().infcx.try(|| {
                eq_tys(self, a.ty, b.ty).then(|| {
                    Ok(ty::mt {ty: a.ty, mutbl: m})
                })
            }).or_else(|e| Err(e))
          }
        }
    }

    fn contratys(&self, a: ty::t, b: ty::t) -> cres<ty::t> {
        Glb(*self.get_ref()).tys(a, b)
    }

    fn purities(&self, a: Purity, b: Purity) -> cres<Purity> {
        match (a, b) {
          (UnsafeFn, _) | (_, UnsafeFn) => Ok(UnsafeFn),
          (ImpureFn, _) | (_, ImpureFn) => Ok(ImpureFn),
          (ExternFn, ExternFn) => Ok(ExternFn),
        }
    }

    fn oncenesses(&self, a: Onceness, b: Onceness) -> cres<Onceness> {
        match (a, b) {
            (Once, _) | (_, Once) => Ok(Once),
            (Many, Many) => Ok(Many)
        }
    }

    fn bounds(&self, a: BuiltinBounds, b: BuiltinBounds) -> cres<BuiltinBounds> {
        // More bounds is a subtype of fewer bounds, so
        // the LUB (mutual supertype) is the intersection.
        Ok(a.intersection(b))
    }

    fn contraregions(&self, a: ty::Region, b: ty::Region)
                    -> cres<ty::Region> {
        return Glb(*self.get_ref()).regions(a, b);
    }

    fn regions(&self, a: ty::Region, b: ty::Region) -> cres<ty::Region> {
        debug!("{}.regions({:?}, {:?})",
               self.tag(),
               a.inf_str(self.get_ref().infcx),
               b.inf_str(self.get_ref().infcx));

        Ok(self.get_ref().infcx.region_vars.lub_regions(Subtype(self.get_ref().trace), a, b))
    }

    fn fn_sigs(&self, a: &ty::FnSig, b: &ty::FnSig) -> cres<ty::FnSig> {
        // Note: this is a subtle algorithm.  For a full explanation,
        // please see the large comment in `region_inference.rs`.

        // Take a snapshot.  We'll never roll this back, but in later
        // phases we do want to be able to examine "all bindings that
        // were created as part of this type comparison", and making a
        // snapshot is a convenient way to do that.
        let snapshot = self.get_ref().infcx.region_vars.start_snapshot();

        // Instantiate each bound region with a fresh region variable.
        let (a_with_fresh, a_map) =
            self.get_ref().infcx.replace_bound_regions_with_fresh_regions(
                self.get_ref().trace, a);
        let (b_with_fresh, _) =
            self.get_ref().infcx.replace_bound_regions_with_fresh_regions(
                self.get_ref().trace, b);

        // Collect constraints.
        let sig0 = if_ok!(super_fn_sigs(self, &a_with_fresh, &b_with_fresh));
        debug!("sig0 = {}", sig0.inf_str(self.get_ref().infcx));

        // Generalize the regions appearing in sig0 if possible
        let new_vars =
            self.get_ref().infcx.region_vars.vars_created_since_snapshot(snapshot);
        let sig1 =
            fold_regions_in_sig(
                self.get_ref().infcx.tcx,
                &sig0,
                |r| generalize_region(self, snapshot, new_vars,
                                      sig0.binder_id, &a_map, r));
        return Ok(sig1);

        fn generalize_region(this: &Lub,
                             snapshot: uint,
                             new_vars: &[RegionVid],
                             new_scope: NodeId,
                             a_map: &HashMap<ty::BoundRegion, ty::Region>,
                             r0: ty::Region)
                             -> ty::Region {
            // Regions that pre-dated the LUB computation stay as they are.
            if !is_var_in_set(new_vars, r0) {
                assert!(!r0.is_bound());
                debug!("generalize_region(r0={:?}): not new variable", r0);
                return r0;
            }

            let tainted = this.get_ref().infcx.region_vars.tainted(snapshot, r0);

            // Variables created during LUB computation which are
            // *related* to regions that pre-date the LUB computation
            // stay as they are.
            if !tainted.iter().all(|r| is_var_in_set(new_vars, *r)) {
                debug!("generalize_region(r0={:?}): \
                        non-new-variables found in {:?}",
                       r0, tainted);
                assert!(!r0.is_bound());
                return r0;
            }

            // Otherwise, the variable must be associated with at
            // least one of the variables representing bound regions
            // in both A and B.  Replace the variable with the "first"
            // bound region from A that we find it to be associated
            // with.
            for (a_br, a_r) in a_map.iter() {
                if tainted.iter().any(|x| x == a_r) {
                    debug!("generalize_region(r0={:?}): \
                            replacing with {:?}, tainted={:?}",
                           r0, *a_br, tainted);
                    return ty::ReLateBound(new_scope, *a_br);
                }
            }

            this.get_ref().infcx.tcx.sess.span_bug(
                this.get_ref().trace.origin.span(),
                format!("Region {:?} is not associated with \
                        any bound region from A!", r0))
        }
    }

    fn tys(&self, a: ty::t, b: ty::t) -> cres<ty::t> {
        super_lattice_tys(self, a, b)
    }
}
