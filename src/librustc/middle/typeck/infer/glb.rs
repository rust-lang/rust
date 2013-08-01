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
use middle::typeck::infer::combine::*;
use middle::typeck::infer::lattice::*;
use middle::typeck::infer::lub::Lub;
use middle::typeck::infer::sub::Sub;
use middle::typeck::infer::to_str::InferStr;
use middle::typeck::infer::{cres, InferCtxt};
use middle::typeck::infer::{TypeTrace, Subtype};
use middle::typeck::infer::fold_regions_in_sig;
use middle::typeck::isr_alist;
use syntax::ast;
use syntax::ast::{Many, Once, extern_fn, impure_fn, m_const, m_imm, m_mutbl};
use syntax::ast::{unsafe_fn};
use syntax::ast::{Onceness, purity};
use syntax::abi::AbiSet;
use util::common::{indenter};
use util::ppaux::mt_to_str;

use extra::list;

pub struct Glb(CombineFields);  // "greatest lower bound" (common subtype)

impl Combine for Glb {
    fn infcx(&self) -> @mut InferCtxt { self.infcx }
    fn tag(&self) -> ~str { ~"glb" }
    fn a_is_expected(&self) -> bool { self.a_is_expected }
    fn trace(&self) -> TypeTrace { self.trace }

    fn sub(&self) -> Sub { Sub(**self) }
    fn lub(&self) -> Lub { Lub(**self) }
    fn glb(&self) -> Glb { Glb(**self) }

    fn mts(&self, a: &ty::mt, b: &ty::mt) -> cres<ty::mt> {
        let tcx = self.infcx.tcx;

        debug!("%s.mts(%s, %s)",
               self.tag(),
               mt_to_str(tcx, a),
               mt_to_str(tcx, b));

        match (a.mutbl, b.mutbl) {
          // If one side or both is mut, then the GLB must use
          // the precise type from the mut side.
          (m_mutbl, m_const) => {
            Sub(**self).tys(a.ty, b.ty).chain(|_t| {
                Ok(ty::mt {ty: a.ty, mutbl: m_mutbl})
            })
          }
          (m_const, m_mutbl) => {
            Sub(**self).tys(b.ty, a.ty).chain(|_t| {
                Ok(ty::mt {ty: b.ty, mutbl: m_mutbl})
            })
          }
          (m_mutbl, m_mutbl) => {
            eq_tys(self, a.ty, b.ty).then(|| {
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

    fn contratys(&self, a: ty::t, b: ty::t) -> cres<ty::t> {
        Lub(**self).tys(a, b)
    }

    fn purities(&self, a: purity, b: purity) -> cres<purity> {
        match (a, b) {
          (extern_fn, _) | (_, extern_fn) => Ok(extern_fn),
          (impure_fn, _) | (_, impure_fn) => Ok(impure_fn),
          (unsafe_fn, unsafe_fn) => Ok(unsafe_fn)
        }
    }

    fn oncenesses(&self, a: Onceness, b: Onceness) -> cres<Onceness> {
        match (a, b) {
            (Many, _) | (_, Many) => Ok(Many),
            (Once, Once) => Ok(Once)
        }
    }

    fn bounds(&self, a: BuiltinBounds, b: BuiltinBounds) -> cres<BuiltinBounds> {
        // More bounds is a subtype of fewer bounds, so
        // the GLB (mutual subtype) is the union.
        Ok(a.union(b))
    }

    fn regions(&self, a: ty::Region, b: ty::Region) -> cres<ty::Region> {
        debug!("%s.regions(%?, %?)",
               self.tag(),
               a.inf_str(self.infcx),
               b.inf_str(self.infcx));

        Ok(self.infcx.region_vars.glb_regions(Subtype(self.trace), a, b))
    }

    fn contraregions(&self, a: ty::Region, b: ty::Region)
                    -> cres<ty::Region> {
        Lub(**self).regions(a, b)
    }

    fn tys(&self, a: ty::t, b: ty::t) -> cres<ty::t> {
        super_lattice_tys(self, a, b)
    }

    // Traits please (FIXME: #2794):

    fn flds(&self, a: ty::field, b: ty::field) -> cres<ty::field> {
        super_flds(self, a, b)
    }

    fn vstores(&self, vk: ty::terr_vstore_kind,
               a: ty::vstore, b: ty::vstore) -> cres<ty::vstore> {
        super_vstores(self, vk, a, b)
    }

    fn trait_stores(&self,
                    vk: ty::terr_vstore_kind,
                    a: ty::TraitStore,
                    b: ty::TraitStore)
                 -> cres<ty::TraitStore> {
        super_trait_stores(self, vk, a, b)
    }

    fn args(&self, a: ty::t, b: ty::t) -> cres<ty::t> {
        super_args(self, a, b)
    }

    fn fn_sigs(&self, a: &ty::FnSig, b: &ty::FnSig) -> cres<ty::FnSig> {
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
                self.trace, a);
        let a_vars = var_ids(self, a_isr);
        let (b_with_fresh, b_isr) =
            self.infcx.replace_bound_regions_with_fresh_regions(
                self.trace, b);
        let b_vars = var_ids(self, b_isr);

        // Collect constraints.
        let sig0 = if_ok!(super_fn_sigs(self, &a_with_fresh, &b_with_fresh));
        debug!("sig0 = %s", sig0.inf_str(self.infcx));

        // Generalize the regions appearing in fn_ty0 if possible
        let new_vars =
            self.infcx.region_vars.vars_created_since_snapshot(snapshot);
        let sig1 =
            fold_regions_in_sig(
                self.infcx.tcx,
                &sig0,
                |r, _in_fn| generalize_region(self, snapshot,
                                              new_vars, a_isr, a_vars, b_vars,
                                              r));
        debug!("sig1 = %s", sig1.inf_str(self.infcx));
        return Ok(sig1);

        fn generalize_region(this: &Glb,
                             snapshot: uint,
                             new_vars: &[RegionVid],
                             a_isr: isr_alist,
                             a_vars: &[RegionVid],
                             b_vars: &[RegionVid],
                             r0: ty::Region) -> ty::Region {
            if !is_var_in_set(new_vars, r0) {
                return r0;
            }

            let tainted = this.infcx.region_vars.tainted(snapshot, r0);

            let mut a_r = None;
            let mut b_r = None;
            let mut only_new_vars = true;
            foreach r in tainted.iter() {
                if is_var_in_set(a_vars, *r) {
                    if a_r.is_some() {
                        return fresh_bound_variable(this);
                    } else {
                        a_r = Some(*r);
                    }
                } else if is_var_in_set(b_vars, *r) {
                    if b_r.is_some() {
                        return fresh_bound_variable(this);
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
                // like: &fn(fn(&a)) and fn(fn(&b)), where a and b are
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
                return rev_lookup(this, a_isr, a_r.get());
            } else if a_r.is_none() && b_r.is_none() {
                // Not related to bound variables from either fn:
                return r0;
            } else {
                // Other:
                return fresh_bound_variable(this);
            }
        }

        fn rev_lookup(this: &Glb,
                      a_isr: isr_alist,
                      r: ty::Region) -> ty::Region
        {
            for list::each(a_isr) |pair| {
                let (a_br, a_r) = *pair;
                if a_r == r {
                    return ty::re_bound(a_br);
                }
            }

            this.infcx.tcx.sess.span_bug(
                this.trace.origin.span(),
                fmt!("could not find original bound region for %?", r));
        }

        fn fresh_bound_variable(this: &Glb) -> ty::Region {
            this.infcx.region_vars.new_bound()
        }
    }

    fn sigils(&self, p1: ast::Sigil, p2: ast::Sigil) -> cres<ast::Sigil> {
        super_sigils(self, p1, p2)
    }

    fn abis(&self, p1: AbiSet, p2: AbiSet) -> cres<AbiSet> {
        super_abis(self, p1, p2)
    }

    fn bare_fn_tys(&self, a: &ty::BareFnTy,
                   b: &ty::BareFnTy) -> cres<ty::BareFnTy> {
        super_bare_fn_tys(self, a, b)
    }

    fn closure_tys(&self, a: &ty::ClosureTy,
                   b: &ty::ClosureTy) -> cres<ty::ClosureTy> {
        super_closure_tys(self, a, b)
    }

    fn substs(&self,
              generics: &ty::Generics,
              as_: &ty::substs,
              bs: &ty::substs) -> cres<ty::substs> {
        super_substs(self, generics, as_, bs)
    }

    fn tps(&self, as_: &[ty::t], bs: &[ty::t]) -> cres<~[ty::t]> {
        super_tps(self, as_, bs)
    }

    fn self_tys(&self, a: Option<ty::t>, b: Option<ty::t>)
               -> cres<Option<ty::t>> {
        super_self_tys(self, a, b)
    }

    fn trait_refs(&self, a: &ty::TraitRef, b: &ty::TraitRef) -> cres<ty::TraitRef> {
        super_trait_refs(self, a, b)
    }
}
