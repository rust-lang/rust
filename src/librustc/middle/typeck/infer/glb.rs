// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use middle::ty::RegionVid;
use middle::ty;
use middle::typeck::infer::combine::*;
use middle::typeck::infer::glb::Glb;
use middle::typeck::infer::lattice::*;
use middle::typeck::infer::lub::Lub;
use middle::typeck::infer::sub::Sub;
use middle::typeck::infer::to_str::InferStr;
use middle::typeck::isr_alist;
use syntax::ast::{Many, Once, extern_fn, impure_fn, m_const, m_imm, m_mutbl};
use syntax::ast::{noreturn, pure_fn, ret_style, return_val, unsafe_fn};
use util::ppaux::mt_to_str;

use std::list;

enum Glb = CombineFields;  // "greatest lower bound" (common subtype)

impl Glb: Combine {
    fn infcx() -> @InferCtxt { self.infcx }
    fn tag() -> ~str { ~"glb" }
    fn a_is_expected() -> bool { self.a_is_expected }
    fn span() -> span { self.span }

    fn sub() -> Sub { Sub(*self) }
    fn lub() -> Lub { Lub(*self) }
    fn glb() -> Glb { Glb(*self) }

    fn mts(a: ty::mt, b: ty::mt) -> cres<ty::mt> {
        let tcx = self.infcx.tcx;

        debug!("%s.mts(%s, %s)",
               self.tag(),
               mt_to_str(tcx, a),
               mt_to_str(tcx, b));

        match (a.mutbl, b.mutbl) {
          // If one side or both is mut, then the GLB must use
          // the precise type from the mut side.
          (m_mutbl, m_const) => {
            Sub(*self).tys(a.ty, b.ty).chain(|_t| {
                Ok(ty::mt {ty: a.ty, mutbl: m_mutbl})
            })
          }
          (m_const, m_mutbl) => {
            Sub(*self).tys(b.ty, a.ty).chain(|_t| {
                Ok(ty::mt {ty: b.ty, mutbl: m_mutbl})
            })
          }
          (m_mutbl, m_mutbl) => {
            eq_tys(&self, a.ty, b.ty).then(|| {
                Ok(ty::mt {ty: a.ty, mutbl: m_mutbl})
            })
          }

          // If one side or both is immutable, we can use the GLB of
          // both sides but mutbl must be `m_imm`.
          (m_imm, m_const) |
          (m_const, m_imm) |
          (m_imm, m_imm) => {
            self.tys(a.ty, b.ty).chain(|t| {
                Ok(ty::mt {ty: t, mutbl: m_imm})
            })
          }

          // If both sides are const, then we can use GLB of both
          // sides and mutbl of only `m_const`.
          (m_const, m_const) => {
            self.tys(a.ty, b.ty).chain(|t| {
                Ok(ty::mt {ty: t, mutbl: m_const})
            })
          }

          // There is no mutual subtype of these combinations.
          (m_mutbl, m_imm) |
          (m_imm, m_mutbl) => {
              Err(ty::terr_mutability)
          }
        }
    }

    fn contratys(a: ty::t, b: ty::t) -> cres<ty::t> {
        Lub(*self).tys(a, b)
    }

    fn purities(a: purity, b: purity) -> cres<purity> {
        match (a, b) {
          (pure_fn, _) | (_, pure_fn) => Ok(pure_fn),
          (extern_fn, _) | (_, extern_fn) => Ok(extern_fn),
          (impure_fn, _) | (_, impure_fn) => Ok(impure_fn),
          (unsafe_fn, unsafe_fn) => Ok(unsafe_fn)
        }
    }

    fn oncenesses(a: Onceness, b: Onceness) -> cres<Onceness> {
        match (a, b) {
            (Many, _) | (_, Many) => Ok(Many),
            (Once, Once) => Ok(Once)
        }
    }

    fn regions(a: ty::Region, b: ty::Region) -> cres<ty::Region> {
        debug!("%s.regions(%?, %?)",
               self.tag(),
               a.inf_str(self.infcx),
               b.inf_str(self.infcx));

        do indent {
            self.infcx.region_vars.glb_regions(self.span, a, b)
        }
    }

    fn contraregions(a: ty::Region, b: ty::Region) -> cres<ty::Region> {
        Lub(*self).regions(a, b)
    }

    fn tys(a: ty::t, b: ty::t) -> cres<ty::t> {
        super_lattice_tys(&self, a, b)
    }

    // Traits please (FIXME: #2794):

    fn flds(a: ty::field, b: ty::field) -> cres<ty::field> {
        super_flds(&self, a, b)
    }

    fn vstores(vk: ty::terr_vstore_kind,
               a: ty::vstore, b: ty::vstore) -> cres<ty::vstore> {
        super_vstores(&self, vk, a, b)
    }

    fn modes(a: ast::mode, b: ast::mode) -> cres<ast::mode> {
        super_modes(&self, a, b)
    }

    fn args(a: ty::arg, b: ty::arg) -> cres<ty::arg> {
        super_args(&self, a, b)
    }

    fn fn_sigs(a: &ty::FnSig, b: &ty::FnSig) -> cres<ty::FnSig> {
        // Note: this is a subtle algorithm.  For a full explanation,
        // please see the large comment in `region_inference.rs`.

        debug!("%s.fn_sigs(%?, %?)",
               self.tag(), a.inf_str(self.infcx), b.inf_str(self.infcx));
        let _indenter = indenter();

        // Take a snapshot.  We'll never roll this back, but in later
        // phases we do want to be able to examine "all bindings that
        // were created as part of this type comparison", and making a
        // snapshot is a convenient way to do that.
        let snapshot = self.infcx.region_vars.start_snapshot();

        // Instantiate each bound region with a fresh region variable.
        let (a_with_fresh, a_isr) =
            self.infcx.replace_bound_regions_with_fresh_regions(
                self.span, a);
        let a_vars = var_ids(&self, a_isr);
        let (b_with_fresh, b_isr) =
            self.infcx.replace_bound_regions_with_fresh_regions(
                self.span, b);
        let b_vars = var_ids(&self, b_isr);

        // Collect constraints.
        let sig0 = if_ok!(super_fn_sigs(&self, &a_with_fresh, &b_with_fresh));
        debug!("sig0 = %s", sig0.inf_str(self.infcx));

        // Generalize the regions appearing in fn_ty0 if possible
        let new_vars =
            self.infcx.region_vars.vars_created_since_snapshot(snapshot);
        let sig1 =
            self.infcx.fold_regions_in_sig(
                &sig0,
                |r, _in_fn| generalize_region(&self, snapshot,
                                              new_vars, a_isr, a_vars, b_vars,
                                              r));
        debug!("sig1 = %s", sig1.inf_str(self.infcx));
        return Ok(move sig1);

        fn generalize_region(self: &Glb,
                             snapshot: uint,
                             new_vars: &[RegionVid],
                             a_isr: isr_alist,
                             a_vars: &[RegionVid],
                             b_vars: &[RegionVid],
                             r0: ty::Region) -> ty::Region {
            if !is_var_in_set(new_vars, r0) {
                return r0;
            }

            let tainted = self.infcx.region_vars.tainted(snapshot, r0);

            let mut a_r = None, b_r = None, only_new_vars = true;
            for tainted.each |r| {
                if is_var_in_set(a_vars, *r) {
                    if a_r.is_some() {
                        return fresh_bound_variable(self);
                    } else {
                        a_r = Some(*r);
                    }
                } else if is_var_in_set(b_vars, *r) {
                    if b_r.is_some() {
                        return fresh_bound_variable(self);
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
                // like: fn(fn(&a)) and fn(fn(&b)), where a and b are
                // free, it will return fn(&c) where c = GLB(a,b).  If
                // however this GLB is not defined, then the result is
                // an error, even though something like
                // "fn<X>(fn(&X))" where X is bound would be a
                // subtype of both of those.
                //
                // The problem is that if we were to return a bound
                // variable, we'd be computing a lower-bound, but not
                // necessarily the *greatest* lower-bound.

            if a_r.is_some() && b_r.is_some() && only_new_vars {
                // Related to exactly one bound variable from each fn:
                return rev_lookup(self, a_isr, a_r.get());
            } else if a_r.is_none() && b_r.is_none() {
                // Not related to bound variables from either fn:
                return r0;
            } else {
                // Other:
                return fresh_bound_variable(self);
            }
        }

        fn rev_lookup(self: &Glb,
                      a_isr: isr_alist,
                      r: ty::Region) -> ty::Region
        {
            for list::each(a_isr) |pair| {
                let (a_br, a_r) = *pair;
                if a_r == r {
                    return ty::re_bound(a_br);
                }
            }

            self.infcx.tcx.sess.span_bug(
                self.span,
                fmt!("could not find original bound region for %?", r));
        }

        fn fresh_bound_variable(self: &Glb) -> ty::Region {
            self.infcx.region_vars.new_bound()
        }
    }

    fn protos(p1: ast::Proto, p2: ast::Proto) -> cres<ast::Proto> {
        super_protos(&self, p1, p2)
    }

    fn fns(a: &ty::FnTy, b: &ty::FnTy) -> cres<ty::FnTy> {
        super_fns(&self, a, b)
    }

    fn fn_metas(a: &ty::FnMeta, b: &ty::FnMeta) -> cres<ty::FnMeta> {
        super_fn_metas(&self, a, b)
    }

    fn substs(did: ast::def_id,
              as_: &ty::substs,
              bs: &ty::substs) -> cres<ty::substs> {
        super_substs(&self, did, as_, bs)
    }

    fn tps(as_: &[ty::t], bs: &[ty::t]) -> cres<~[ty::t]> {
        super_tps(&self, as_, bs)
    }

    fn self_tys(a: Option<ty::t>, b: Option<ty::t>) -> cres<Option<ty::t>> {
        super_self_tys(&self, a, b)
    }
}

