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
use middle::ty;
use middle::ty::TyVar;
use middle::typeck::check::regionmanip::replace_late_bound_regions_in_fn_sig;
use middle::typeck::infer::combine::*;
use middle::typeck::infer::{cres, CresCompare};
use middle::typeck::infer::glb::Glb;
use middle::typeck::infer::InferCtxt;
use middle::typeck::infer::lattice::CombineFieldsLatticeMethods;
use middle::typeck::infer::lub::Lub;
use middle::typeck::infer::then;
use middle::typeck::infer::{TypeTrace, Subtype};
use util::common::{indenter};
use util::ppaux::{bound_region_to_string, Repr};

use syntax::ast::{Onceness, FnStyle, MutImmutable, MutMutable};

pub struct Sub<'f>(pub CombineFields<'f>);  // "subtype", "subregion" etc

impl<'f> Sub<'f> {
    pub fn get_ref<'a>(&'a self) -> &'a CombineFields<'f> { let Sub(ref v) = *self; v }
}

impl<'f> Combine for Sub<'f> {
    fn infcx<'a>(&'a self) -> &'a InferCtxt<'a> { self.get_ref().infcx }
    fn tag(&self) -> String { "sub".to_string() }
    fn a_is_expected(&self) -> bool { self.get_ref().a_is_expected }
    fn trace(&self) -> TypeTrace { self.get_ref().trace.clone() }

    fn sub<'a>(&'a self) -> Sub<'a> { Sub(self.get_ref().clone()) }
    fn lub<'a>(&'a self) -> Lub<'a> { Lub(self.get_ref().clone()) }
    fn glb<'a>(&'a self) -> Glb<'a> { Glb(self.get_ref().clone()) }

    fn contratys(&self, a: ty::t, b: ty::t) -> cres<ty::t> {
        let opp = CombineFields {
            a_is_expected: !self.get_ref().a_is_expected,
            ..self.get_ref().clone()
        };
        Sub(opp).tys(b, a)
    }

    fn contraregions(&self, a: ty::Region, b: ty::Region)
                    -> cres<ty::Region> {
        let opp = CombineFields {
            a_is_expected: !self.get_ref().a_is_expected,
            ..self.get_ref().clone()
        };
        Sub(opp).regions(b, a)
    }

    fn regions(&self, a: ty::Region, b: ty::Region) -> cres<ty::Region> {
        debug!("{}.regions({}, {})",
               self.tag(),
               a.repr(self.get_ref().infcx.tcx),
               b.repr(self.get_ref().infcx.tcx));
        self.get_ref().infcx.region_vars.make_subregion(Subtype(self.trace()), a, b);
        Ok(a)
    }

    fn mts(&self, a: &ty::mt, b: &ty::mt) -> cres<ty::mt> {
        debug!("mts({} <: {})",
               a.repr(self.get_ref().infcx.tcx),
               b.repr(self.get_ref().infcx.tcx));

        if a.mutbl != b.mutbl {
            return Err(ty::terr_mutability);
        }

        match b.mutbl {
          MutMutable => {
            // If supertype is mut, subtype must match exactly
            // (i.e., invariant if mut):
            eq_tys(self, a.ty, b.ty).then(|| Ok(*a))
          }
          MutImmutable => {
            // Otherwise we can be covariant:
            self.tys(a.ty, b.ty).and_then(|_t| Ok(*a) )
          }
        }
    }

    fn fn_styles(&self, a: FnStyle, b: FnStyle) -> cres<FnStyle> {
        self.lub().fn_styles(a, b).compare(b, || {
            ty::terr_fn_style_mismatch(expected_found(self, a, b))
        })
    }

    fn oncenesses(&self, a: Onceness, b: Onceness) -> cres<Onceness> {
        self.lub().oncenesses(a, b).compare(b, || {
            ty::terr_onceness_mismatch(expected_found(self, a, b))
        })
    }

    fn bounds(&self, a: BuiltinBounds, b: BuiltinBounds) -> cres<BuiltinBounds> {
        // More bounds is a subtype of fewer bounds.
        //
        // e.g., fn:Copy() <: fn(), because the former is a function
        // that only closes over copyable things, but the latter is
        // any function at all.
        if a.contains(b) {
            Ok(a)
        } else {
            Err(ty::terr_builtin_bounds(expected_found(self, a, b)))
        }
    }

    fn tys(&self, a: ty::t, b: ty::t) -> cres<ty::t> {
        debug!("{}.tys({}, {})", self.tag(),
               a.repr(self.get_ref().infcx.tcx), b.repr(self.get_ref().infcx.tcx));
        if a == b { return Ok(a); }
        let _indenter = indenter();
        match (&ty::get(a).sty, &ty::get(b).sty) {
            (&ty::ty_bot, _) => {
                Ok(a)
            }

            (&ty::ty_infer(TyVar(a_id)), &ty::ty_infer(TyVar(b_id))) => {
                if_ok!(self.get_ref().var_sub_var(a_id, b_id));
                Ok(a)
            }
            (&ty::ty_infer(TyVar(a_id)), _) => {
                if_ok!(self.get_ref().var_sub_t(a_id, b));
                Ok(a)
            }
            (_, &ty::ty_infer(TyVar(b_id))) => {
                if_ok!(self.get_ref().t_sub_var(a, b_id));
                Ok(a)
            }

            (_, &ty::ty_bot) => {
                Err(ty::terr_sorts(expected_found(self, a, b)))
            }

            _ => {
                super_tys(self, a, b)
            }
        }
    }

    fn fn_sigs(&self, a: &ty::FnSig, b: &ty::FnSig) -> cres<ty::FnSig> {
        debug!("fn_sigs(a={}, b={})",
               a.repr(self.get_ref().infcx.tcx), b.repr(self.get_ref().infcx.tcx));
        let _indenter = indenter();

        // Rather than checking the subtype relationship between `a` and `b`
        // as-is, we need to do some extra work here in order to make sure
        // that function subtyping works correctly with respect to regions
        //
        // Note: this is a subtle algorithm.  For a full explanation,
        // please see the large comment in `region_inference.rs`.

        // Make a mark so we can examine "all bindings that were
        // created as part of this type comparison".
        let mark = self.get_ref().infcx.region_vars.mark();

        // First, we instantiate each bound region in the subtype with a fresh
        // region variable.
        let (a_sig, _) =
            self.get_ref().infcx.replace_late_bound_regions_with_fresh_regions(
                self.trace(), a);

        // Second, we instantiate each bound region in the supertype with a
        // fresh concrete region.
        let (skol_map, b_sig) = {
            replace_late_bound_regions_in_fn_sig(self.get_ref().infcx.tcx, b, |br| {
                let skol = self.get_ref().infcx.region_vars.new_skolemized(br);
                debug!("Bound region {} skolemized to {:?}",
                       bound_region_to_string(self.get_ref().infcx.tcx, "", false, br),
                       skol);
                skol
            })
        };

        debug!("a_sig={}", a_sig.repr(self.get_ref().infcx.tcx));
        debug!("b_sig={}", b_sig.repr(self.get_ref().infcx.tcx));

        // Compare types now that bound regions have been replaced.
        let sig = if_ok!(super_fn_sigs(self, &a_sig, &b_sig));

        // Presuming type comparison succeeds, we need to check
        // that the skolemized regions do not "leak".
        let new_vars =
            self.get_ref().infcx.region_vars.vars_created_since_mark(mark);
        for (&skol_br, &skol) in skol_map.iter() {
            let tainted = self.get_ref().infcx.region_vars.tainted(mark, skol);
            for tainted_region in tainted.iter() {
                // Each skolemized should only be relatable to itself
                // or new variables:
                match *tainted_region {
                    ty::ReInfer(ty::ReVar(ref vid)) => {
                        if new_vars.iter().any(|x| x == vid) { continue; }
                    }
                    _ => {
                        if *tainted_region == skol { continue; }
                    }
                };

                // A is not as polymorphic as B:
                if self.a_is_expected() {
                    debug!("Not as polymorphic!");
                    return Err(ty::terr_regions_insufficiently_polymorphic(
                            skol_br, *tainted_region));
                } else {
                    debug!("Overly polymorphic!");
                    return Err(ty::terr_regions_overly_polymorphic(
                            skol_br, *tainted_region));
                }
            }
        }

        return Ok(sig);
    }

}
