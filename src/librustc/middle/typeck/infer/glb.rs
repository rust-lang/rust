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
use middle::typeck::infer::lattice::*;
use middle::typeck::infer::lub::Lub;
use middle::typeck::infer::sub::Sub;
use middle::typeck::infer::{cres, InferCtxt};
use middle::typeck::infer::{TypeTrace, Subtype};
use middle::typeck::infer::fold_regions_in_sig;
use middle::typeck::infer::region_inference::RegionMark;
use syntax::ast::{Many, Once, MutImmutable, MutMutable};
use syntax::ast::{NormalFn, UnsafeFn, NodeId};
use syntax::ast::{Onceness, FnStyle};
use std::collections::HashMap;
use util::common::{indenter};
use util::ppaux::mt_to_string;
use util::ppaux::Repr;

pub struct Glb<'f>(pub CombineFields<'f>);  // "greatest lower bound" (common subtype)

impl<'f> Glb<'f> {
    pub fn get_ref<'a>(&'a self) -> &'a CombineFields<'f> { let Glb(ref v) = *self; v }
}

impl<'f> Combine for Glb<'f> {
    fn infcx<'a>(&'a self) -> &'a InferCtxt<'a> { self.get_ref().infcx }
    fn tag(&self) -> String { "glb".to_string() }
    fn a_is_expected(&self) -> bool { self.get_ref().a_is_expected }
    fn trace(&self) -> TypeTrace { self.get_ref().trace.clone() }

    fn sub<'a>(&'a self) -> Sub<'a> { Sub(self.get_ref().clone()) }
    fn lub<'a>(&'a self) -> Lub<'a> { Lub(self.get_ref().clone()) }
    fn glb<'a>(&'a self) -> Glb<'a> { Glb(self.get_ref().clone()) }

    fn mts(&self, a: &ty::mt, b: &ty::mt) -> cres<ty::mt> {
        let tcx = self.get_ref().infcx.tcx;

        debug!("{}.mts({}, {})",
               self.tag(),
               mt_to_string(tcx, a),
               mt_to_string(tcx, b));

        match (a.mutbl, b.mutbl) {
          // If one side or both is mut, then the GLB must use
          // the precise type from the mut side.
          (MutMutable, MutMutable) => {
            eq_tys(self, a.ty, b.ty).then(|| {
                Ok(ty::mt {ty: a.ty, mutbl: MutMutable})
            })
          }

          // If one side or both is immutable, we can use the GLB of
          // both sides but mutbl must be `MutImmutable`.
          (MutImmutable, MutImmutable) => {
            self.tys(a.ty, b.ty).and_then(|t| {
                Ok(ty::mt {ty: t, mutbl: MutImmutable})
            })
          }

          // There is no mutual subtype of these combinations.
          (MutMutable, MutImmutable) |
          (MutImmutable, MutMutable) => {
              Err(ty::terr_mutability)
          }
        }
    }

    fn contratys(&self, a: ty::t, b: ty::t) -> cres<ty::t> {
        self.lub().tys(a, b)
    }

    fn fn_styles(&self, a: FnStyle, b: FnStyle) -> cres<FnStyle> {
        match (a, b) {
          (NormalFn, _) | (_, NormalFn) => Ok(NormalFn),
          (UnsafeFn, UnsafeFn) => Ok(UnsafeFn)
        }
    }

    fn oncenesses(&self, a: Onceness, b: Onceness) -> cres<Onceness> {
        match (a, b) {
            (Many, _) | (_, Many) => Ok(Many),
            (Once, Once) => Ok(Once)
        }
    }

    fn builtin_bounds(&self,
                      a: ty::BuiltinBounds,
                      b: ty::BuiltinBounds)
                      -> cres<ty::BuiltinBounds> {
        // More bounds is a subtype of fewer bounds, so
        // the GLB (mutual subtype) is the union.
        Ok(a.union(b))
    }

    fn regions(&self, a: ty::Region, b: ty::Region) -> cres<ty::Region> {
        debug!("{}.regions({:?}, {:?})",
               self.tag(),
               a.repr(self.get_ref().infcx.tcx),
               b.repr(self.get_ref().infcx.tcx));

        Ok(self.get_ref().infcx.region_vars.glb_regions(Subtype(self.trace()), a, b))
    }

    fn contraregions(&self, a: ty::Region, b: ty::Region)
                    -> cres<ty::Region> {
        self.lub().regions(a, b)
    }

    fn tys(&self, a: ty::t, b: ty::t) -> cres<ty::t> {
        super_lattice_tys(self, a, b)
    }

    fn fn_sigs(&self, a: &ty::FnSig, b: &ty::FnSig) -> cres<ty::FnSig> {
        // Note: this is a subtle algorithm.  For a full explanation,
        // please see the large comment in `region_inference.rs`.

        debug!("{}.fn_sigs({:?}, {:?})",
               self.tag(), a.repr(self.get_ref().infcx.tcx), b.repr(self.get_ref().infcx.tcx));
        let _indenter = indenter();

        // Make a mark so we can examine "all bindings that were
        // created as part of this type comparison".
        let mark = self.get_ref().infcx.region_vars.mark();

        // Instantiate each bound region with a fresh region variable.
        let (a_with_fresh, a_map) =
            self.get_ref().infcx.replace_late_bound_regions_with_fresh_regions(
                self.trace(), a);
        let a_vars = var_ids(self, &a_map);
        let (b_with_fresh, b_map) =
            self.get_ref().infcx.replace_late_bound_regions_with_fresh_regions(
                self.trace(), b);
        let b_vars = var_ids(self, &b_map);

        // Collect constraints.
        let sig0 = if_ok!(super_fn_sigs(self, &a_with_fresh, &b_with_fresh));
        debug!("sig0 = {}", sig0.repr(self.get_ref().infcx.tcx));

        // Generalize the regions appearing in fn_ty0 if possible
        let new_vars =
            self.get_ref().infcx.region_vars.vars_created_since_mark(mark);
        let sig1 =
            fold_regions_in_sig(
                self.get_ref().infcx.tcx,
                &sig0,
                |r| {
                generalize_region(self,
                                  mark,
                                  new_vars.as_slice(),
                                  sig0.binder_id,
                                  &a_map,
                                  a_vars.as_slice(),
                                  b_vars.as_slice(),
                                  r)
            });
        debug!("sig1 = {}", sig1.repr(self.get_ref().infcx.tcx));
        return Ok(sig1);

        fn generalize_region(this: &Glb,
                             mark: RegionMark,
                             new_vars: &[RegionVid],
                             new_binder_id: NodeId,
                             a_map: &HashMap<ty::BoundRegion, ty::Region>,
                             a_vars: &[RegionVid],
                             b_vars: &[RegionVid],
                             r0: ty::Region) -> ty::Region {
            if !is_var_in_set(new_vars, r0) {
                assert!(!r0.is_bound());
                return r0;
            }

            let tainted = this.get_ref().infcx.region_vars.tainted(mark, r0);

            let mut a_r = None;
            let mut b_r = None;
            let mut only_new_vars = true;
            for r in tainted.iter() {
                if is_var_in_set(a_vars, *r) {
                    if a_r.is_some() {
                        return fresh_bound_variable(this, new_binder_id);
                    } else {
                        a_r = Some(*r);
                    }
                } else if is_var_in_set(b_vars, *r) {
                    if b_r.is_some() {
                        return fresh_bound_variable(this, new_binder_id);
                    } else {
                        b_r = Some(*r);
                    }
                } else if !is_var_in_set(new_vars, *r) {
                    only_new_vars = false;
                }
            }

            // NB---I do not believe this algorithm computes
            // (necessarily) the GLB.  As written it can
            // spuriously fail.  In particular, if there is a case
            // like: |fn(&a)| and fn(fn(&b)), where a and b are
            // free, it will return fn(&c) where c = GLB(a,b).  If
            // however this GLB is not defined, then the result is
            // an error, even though something like
            // "fn<X>(fn(&X))" where X is bound would be a
            // subtype of both of those.
            //
            // The problem is that if we were to return a bound
            // variable, we'd be computing a lower-bound, but not
            // necessarily the *greatest* lower-bound.
            //
            // Unfortunately, this problem is non-trivial to solve,
            // because we do not know at the time of computing the GLB
            // whether a GLB(a,b) exists or not, because we haven't
            // run region inference (or indeed, even fully computed
            // the region hierarchy!). The current algorithm seems to
            // works ok in practice.

            if a_r.is_some() && b_r.is_some() && only_new_vars {
                // Related to exactly one bound variable from each fn:
                return rev_lookup(this, a_map, new_binder_id, a_r.unwrap());
            } else if a_r.is_none() && b_r.is_none() {
                // Not related to bound variables from either fn:
                assert!(!r0.is_bound());
                return r0;
            } else {
                // Other:
                return fresh_bound_variable(this, new_binder_id);
            }
        }

        fn rev_lookup(this: &Glb,
                      a_map: &HashMap<ty::BoundRegion, ty::Region>,
                      new_binder_id: NodeId,
                      r: ty::Region) -> ty::Region
        {
            for (a_br, a_r) in a_map.iter() {
                if *a_r == r {
                    return ty::ReLateBound(new_binder_id, *a_br);
                }
            }
            this.get_ref().infcx.tcx.sess.span_bug(
                this.get_ref().trace.origin.span(),
                format!("could not find original bound region for {:?}",
                        r).as_slice())
        }

        fn fresh_bound_variable(this: &Glb, binder_id: NodeId) -> ty::Region {
            this.get_ref().infcx.region_vars.new_bound(binder_id)
        }
    }
}
