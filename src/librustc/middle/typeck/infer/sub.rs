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
use middle::typeck::infer::equate::Equate;
use middle::typeck::infer::glb::Glb;
use middle::typeck::infer::InferCtxt;
use middle::typeck::infer::lub::Lub;
use middle::typeck::infer::{TypeTrace, Subtype};
use middle::typeck::infer::type_variable::{SubtypeOf, SupertypeOf};
use util::common::{indenter};
use util::ppaux::{bound_region_to_string, Repr};

use syntax::ast::{Onceness, FnStyle, MutImmutable, MutMutable};


/// "Greatest lower bound" (common subtype)
pub struct Sub<'f> {
    fields: CombineFields<'f>
}

#[allow(non_snake_case_functions)]
pub fn Sub<'f>(cf: CombineFields<'f>) -> Sub<'f> {
    Sub { fields: cf }
}

impl<'f> Combine for Sub<'f> {
    fn infcx<'a>(&'a self) -> &'a InferCtxt<'a> { self.fields.infcx }
    fn tag(&self) -> String { "sub".to_string() }
    fn a_is_expected(&self) -> bool { self.fields.a_is_expected }
    fn trace(&self) -> TypeTrace { self.fields.trace.clone() }

    fn equate<'a>(&'a self) -> Equate<'a> { Equate(self.fields.clone()) }
    fn sub<'a>(&'a self) -> Sub<'a> { Sub(self.fields.clone()) }
    fn lub<'a>(&'a self) -> Lub<'a> { Lub(self.fields.clone()) }
    fn glb<'a>(&'a self) -> Glb<'a> { Glb(self.fields.clone()) }

    fn contratys(&self, a: ty::t, b: ty::t) -> cres<ty::t> {
        Sub(self.fields.switch_expected()).tys(b, a)
    }

    fn contraregions(&self, a: ty::Region, b: ty::Region)
                     -> cres<ty::Region> {
                         let opp = CombineFields {
                             a_is_expected: !self.fields.a_is_expected,
                             ..self.fields.clone()
                         };
                         Sub(opp).regions(b, a)
                     }

    fn regions(&self, a: ty::Region, b: ty::Region) -> cres<ty::Region> {
        debug!("{}.regions({}, {})",
               self.tag(),
               a.repr(self.fields.infcx.tcx),
               b.repr(self.fields.infcx.tcx));
        self.fields.infcx.region_vars.make_subregion(Subtype(self.trace()), a, b);
        Ok(a)
    }

    fn mts(&self, a: &ty::mt, b: &ty::mt) -> cres<ty::mt> {
        debug!("mts({} <: {})",
               a.repr(self.fields.infcx.tcx),
               b.repr(self.fields.infcx.tcx));

        if a.mutbl != b.mutbl {
            return Err(ty::terr_mutability);
        }

        match b.mutbl {
            MutMutable => {
                // If supertype is mut, subtype must match exactly
                // (i.e., invariant if mut):
                try!(self.equate().tys(a.ty, b.ty));
            }
            MutImmutable => {
                // Otherwise we can be covariant:
                try!(self.tys(a.ty, b.ty));
            }
        }

        Ok(*a) // return is meaningless in sub, just return *a
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

    fn builtin_bounds(&self, a: BuiltinBounds, b: BuiltinBounds)
                      -> cres<BuiltinBounds> {
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
               a.repr(self.fields.infcx.tcx), b.repr(self.fields.infcx.tcx));
        if a == b { return Ok(a); }

        let infcx = self.fields.infcx;
        let a = infcx.type_variables.borrow().replace_if_possible(a);
        let b = infcx.type_variables.borrow().replace_if_possible(b);
        match (&ty::get(a).sty, &ty::get(b).sty) {
            (&ty::ty_bot, _) => {
                Ok(a)
            }

            (&ty::ty_infer(TyVar(a_id)), &ty::ty_infer(TyVar(b_id))) => {
                infcx.type_variables
                    .borrow_mut()
                    .relate_vars(a_id, SubtypeOf, b_id);
                Ok(a)
            }
            // The vec/str check here and below is so that we don't unify
            // T with [T], this is necessary so we reflect subtyping of references
            // (&T does not unify with &[T]) where that in turn is to reflect
            // the historical non-typedness of [T].
            (&ty::ty_infer(TyVar(_)), &ty::ty_str) |
            (&ty::ty_infer(TyVar(_)), &ty::ty_vec(_, None)) => {
                Err(ty::terr_sorts(expected_found(self, a, b)))
            }
            (&ty::ty_infer(TyVar(a_id)), _) => {
                try!(self.fields
                       .switch_expected()
                       .instantiate(b, SupertypeOf, a_id));
                Ok(a)
            }

            (&ty::ty_str, &ty::ty_infer(TyVar(_))) |
            (&ty::ty_vec(_, None), &ty::ty_infer(TyVar(_))) => {
                Err(ty::terr_sorts(expected_found(self, a, b)))
            }
            (_, &ty::ty_infer(TyVar(b_id))) => {
                try!(self.fields.instantiate(a, SubtypeOf, b_id));
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
               a.repr(self.fields.infcx.tcx), b.repr(self.fields.infcx.tcx));
        let _indenter = indenter();

        // Rather than checking the subtype relationship between `a` and `b`
        // as-is, we need to do some extra work here in order to make sure
        // that function subtyping works correctly with respect to regions
        //
        // Note: this is a subtle algorithm.  For a full explanation,
        // please see the large comment in `region_inference.rs`.

        // Make a mark so we can examine "all bindings that were
        // created as part of this type comparison".
        let mark = self.fields.infcx.region_vars.mark();

        // First, we instantiate each bound region in the subtype with a fresh
        // region variable.
        let (a_sig, _) =
            self.fields.infcx.replace_late_bound_regions_with_fresh_regions(
                self.trace(), a);

        // Second, we instantiate each bound region in the supertype with a
        // fresh concrete region.
        let (skol_map, b_sig) = {
            replace_late_bound_regions_in_fn_sig(self.fields.infcx.tcx, b, |br| {
                let skol = self.fields.infcx.region_vars.new_skolemized(br);
                debug!("Bound region {} skolemized to {:?}",
                       bound_region_to_string(self.fields.infcx.tcx, "", false, br),
                       skol);
                skol
            })
        };

        debug!("a_sig={}", a_sig.repr(self.fields.infcx.tcx));
        debug!("b_sig={}", b_sig.repr(self.fields.infcx.tcx));

        // Compare types now that bound regions have been replaced.
        let sig = try!(super_fn_sigs(self, &a_sig, &b_sig));

        // Presuming type comparison succeeds, we need to check
        // that the skolemized regions do not "leak".
        let new_vars =
            self.fields.infcx.region_vars.vars_created_since_mark(mark);
        for (&skol_br, &skol) in skol_map.iter() {
            let tainted = self.fields.infcx.region_vars.tainted(mark, skol);
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

