// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ______________________________________________________________________
// Type combining
//
// There are three type combiners: sub, lub, and glb.  Each implements
// the trait `combine` and contains methods for combining two
// instances of various things and yielding a new instance.  These
// combiner methods always yield a `result<T>`---failure is propagated
// upward using `chain()` methods.
//
// There is a lot of common code for these operations, which is
// abstracted out into functions named `super_X()` which take a combiner
// instance as the first parameter.  This would be better implemented
// using traits.  For this system to work properly, you should not
// call the `super_X(foo, ...)` functions directly, but rather call
// `foo.X(...)`.  The implementation of `X()` can then choose to delegate
// to the `super` routine or to do other things.
// (FIXME (#2794): revise this paragraph once default methods in traits
// are working.)
//
// In reality, the sub operation is rather different from lub/glb, but
// they are combined into one trait to avoid duplication (they used to
// be separate but there were many bugs because there were two copies
// of most routines).
//
// The differences are:
//
// - when making two things have a sub relationship, the order of the
//   arguments is significant (a <: b) and the return value of the
//   combine functions is largely irrelevant.  The important thing is
//   whether the action succeeds or fails.  If it succeeds, then side
//   effects have been committed into the type variables.
//
// - for GLB/LUB, the order of arguments is not significant (GLB(a,b) ==
//   GLB(b,a)) and the return value is important (it is the GLB).  Of
//   course GLB/LUB may also have side effects.
//
// Contravariance
//
// When you are relating two things which have a contravariant
// relationship, you should use `contratys()` or `contraregions()`,
// rather than inversing the order of arguments!  This is necessary
// because the order of arguments is not relevant for LUB and GLB.  It
// is also useful to track which value is the "expected" value in
// terms of error reporting, although we do not do that properly right
// now.

use core::prelude::*;

use middle::ty::{FloatVar, FnSig, IntVar, TyVar};
use middle::ty::{IntType, UintType, arg, substs};
use middle::ty;
use middle::typeck::infer::glb::Glb;
use middle::typeck::infer::lub::Lub;
use middle::typeck::infer::sub::Sub;
use middle::typeck::infer::to_str::InferStr;
use middle::typeck::infer::{cres, InferCtxt, ures, IntType, UintType};
use util::common::indent;

use core::result::{iter_vec2, map_vec2};
use core::vec;
use syntax::ast::{Onceness, purity};
use syntax::ast;
use syntax::opt_vec;
use syntax::codemap::span;
use syntax::abi::AbiSet;

pub trait Combine {
    fn infcx(&self) -> @mut InferCtxt;
    fn tag(&self) -> ~str;
    fn a_is_expected(&self) -> bool;
    fn span(&self) -> span;

    fn sub(&self) -> Sub;
    fn lub(&self) -> Lub;
    fn glb(&self) -> Glb;

    fn mts(&self, a: &ty::mt, b: &ty::mt) -> cres<ty::mt>;
    fn contratys(&self, a: ty::t, b: ty::t) -> cres<ty::t>;
    fn tys(&self, a: ty::t, b: ty::t) -> cres<ty::t>;
    fn tps(&self, as_: &[ty::t], bs: &[ty::t]) -> cres<~[ty::t]>;
    fn self_tys(&self, a: Option<ty::t>, b: Option<ty::t>)
               -> cres<Option<ty::t>>;
    fn substs(&self, generics: &ty::Generics, as_: &ty::substs,
              bs: &ty::substs) -> cres<ty::substs>;
    fn bare_fn_tys(&self, a: &ty::BareFnTy,
                   b: &ty::BareFnTy) -> cres<ty::BareFnTy>;
    fn closure_tys(&self, a: &ty::ClosureTy,
                   b: &ty::ClosureTy) -> cres<ty::ClosureTy>;
    fn fn_sigs(&self, a: &ty::FnSig, b: &ty::FnSig) -> cres<ty::FnSig>;
    fn flds(&self, a: ty::field, b: ty::field) -> cres<ty::field>;
    fn modes(&self, a: ast::mode, b: ast::mode) -> cres<ast::mode>;
    fn args(&self, a: ty::arg, b: ty::arg) -> cres<ty::arg>;
    fn sigils(&self, p1: ast::Sigil, p2: ast::Sigil) -> cres<ast::Sigil>;
    fn purities(&self, a: purity, b: purity) -> cres<purity>;
    fn abis(&self, a: AbiSet, b: AbiSet) -> cres<AbiSet>;
    fn oncenesses(&self, a: Onceness, b: Onceness) -> cres<Onceness>;
    fn contraregions(&self, a: ty::Region, b: ty::Region)
                  -> cres<ty::Region>;
    fn regions(&self, a: ty::Region, b: ty::Region) -> cres<ty::Region>;
    fn vstores(&self, vk: ty::terr_vstore_kind,
               a: ty::vstore, b: ty::vstore) -> cres<ty::vstore>;
    fn trait_stores(&self,
                    vk: ty::terr_vstore_kind,
                    a: ty::TraitStore,
                    b: ty::TraitStore)
                 -> cres<ty::TraitStore>;
    fn trait_refs(&self, a: &ty::TraitRef, b: &ty::TraitRef) -> cres<ty::TraitRef>;
}

pub struct CombineFields {
    infcx: @mut InferCtxt,
    a_is_expected: bool,
    span: span,
}

pub fn expected_found<C:Combine,T>(
        self: &C, +a: T, +b: T) -> ty::expected_found<T> {
    if self.a_is_expected() {
        ty::expected_found {expected: a, found: b}
    } else {
        ty::expected_found {expected: b, found: a}
    }
}

pub fn eq_tys<C:Combine>(self: &C, a: ty::t, b: ty::t) -> ures {
    let suber = self.sub();
    do self.infcx().try {
        do suber.tys(a, b).chain |_ok| {
            suber.contratys(a, b)
        }.to_ures()
    }
}

pub fn eq_regions<C:Combine>(self: &C, a: ty::Region, b: ty::Region)
                          -> ures {
    debug!("eq_regions(%s, %s)",
           a.inf_str(self.infcx()),
           b.inf_str(self.infcx()));
    let sub = self.sub();
    do indent {
        self.infcx().try(|| {
            do sub.regions(a, b).chain |_r| {
                sub.contraregions(a, b)
            }
        }).chain_err(|e| {
            // substitute a better error, but use the regions
            // found in the original error
            match e {
              ty::terr_regions_does_not_outlive(a1, b1) =>
                Err(ty::terr_regions_not_same(a1, b1)),
              _ => Err(e)
            }
        }).to_ures()
    }
}

pub fn eq_opt_regions<C:Combine>(
    self: &C,
    a: Option<ty::Region>,
    b: Option<ty::Region>) -> cres<Option<ty::Region>> {

    match (a, b) {
      (None, None) => {
        Ok(None)
      }
      (Some(a), Some(b)) => {
        do eq_regions(self, a, b).then {
            Ok(Some(a))
        }
      }
      (_, _) => {
        // If these two substitutions are for the same type (and
        // they should be), then the type should either
        // consistently have a region parameter or not have a
        // region parameter.
        self.infcx().tcx.sess.bug(
            fmt!("substitution a had opt_region %s and \
                  b had opt_region %s",
                 a.inf_str(self.infcx()),
                 b.inf_str(self.infcx())));
      }
    }
}

pub fn super_substs<C:Combine>(
    self: &C, generics: &ty::Generics,
    a: &ty::substs, b: &ty::substs) -> cres<ty::substs> {

    fn relate_region_param<C:Combine>(
        self: &C,
        generics: &ty::Generics,
        a: Option<ty::Region>,
        b: Option<ty::Region>)
        -> cres<Option<ty::Region>>
    {
        match (&generics.region_param, &a, &b) {
          (&None, &None, &None) => {
            Ok(None)
          }
          (&Some(ty::rv_invariant), &Some(a), &Some(b)) => {
            do eq_regions(self, a, b).then {
                Ok(Some(a))
            }
          }
          (&Some(ty::rv_covariant), &Some(a), &Some(b)) => {
            do self.regions(a, b).chain |r| {
                Ok(Some(r))
            }
          }
          (&Some(ty::rv_contravariant), &Some(a), &Some(b)) => {
            do self.contraregions(a, b).chain |r| {
                Ok(Some(r))
            }
          }
          (_, _, _) => {
            // If these two substitutions are for the same type (and
            // they should be), then the type should either
            // consistently have a region parameter or not have a
            // region parameter, and that should match with the
            // polytype.
            self.infcx().tcx.sess.bug(
                fmt!("substitution a had opt_region %s and \
                      b had opt_region %s with variance %?",
                      a.inf_str(self.infcx()),
                      b.inf_str(self.infcx()),
                     generics.region_param));
          }
        }
    }

    do self.tps(a.tps, b.tps).chain |tps| {
        do self.self_tys(a.self_ty, b.self_ty).chain |self_ty| {
            do relate_region_param(self, generics,
                                   a.self_r, b.self_r).chain |self_r|
            {
                Ok(substs {
                    self_r: self_r,
                    self_ty: self_ty,
                    tps: /*bad*/copy tps
                })
            }
        }
    }
}

pub fn super_tps<C:Combine>(
    self: &C, as_: &[ty::t], bs: &[ty::t]) -> cres<~[ty::t]> {

    // Note: type parameters are always treated as *invariant*
    // (otherwise the type system would be unsound).  In the
    // future we could allow type parameters to declare a
    // variance.

    if vec::same_length(as_, bs) {
        iter_vec2(as_, bs, |a, b| {
            eq_tys(self, *a, *b)
        }).then(|| Ok(as_.to_vec()) )
    } else {
        Err(ty::terr_ty_param_size(
            expected_found(self, as_.len(), bs.len())))
    }
}

pub fn super_self_tys<C:Combine>(
    self: &C, a: Option<ty::t>, b: Option<ty::t>) -> cres<Option<ty::t>> {

    match (a, b) {
      (None, None) => {
        Ok(None)
      }
      (Some(a), Some(b)) => {
          // FIXME(#5781) this should be eq_tys
          // eq_tys(self, a, b).then(|| Ok(Some(a)) )
          self.contratys(a, b).chain(|t| Ok(Some(t)))
      }
      (None, Some(_)) |
      (Some(_), None) => {
        // I think it should never happen that we unify two substs and
        // one of them has a self_ty and one doesn't...? I could be
        // wrong about this.
        Err(ty::terr_self_substs)
      }
    }
}

pub fn super_sigils<C:Combine>(
    self: &C, p1: ast::Sigil, p2: ast::Sigil) -> cres<ast::Sigil> {
    if p1 == p2 {
        Ok(p1)
    } else {
        Err(ty::terr_sigil_mismatch(expected_found(self, p1, p2)))
    }
}

pub fn super_flds<C:Combine>(
    self: &C, a: ty::field, b: ty::field) -> cres<ty::field> {

    if a.ident == b.ident {
        self.mts(&a.mt, &b.mt)
            .chain(|mt| Ok(ty::field {ident: a.ident, mt: mt}) )
            .chain_err(|e| Err(ty::terr_in_field(@e, a.ident)) )
    } else {
        Err(ty::terr_record_fields(
            expected_found(self, a.ident, b.ident)))
    }
}

pub fn super_modes<C:Combine>(
    self: &C, a: ast::mode, b: ast::mode)
    -> cres<ast::mode> {

    let tcx = self.infcx().tcx;
    ty::unify_mode(tcx, expected_found(self, a, b))
}

pub fn super_args<C:Combine>(
    self: &C, a: ty::arg, b: ty::arg)
    -> cres<ty::arg> {

    do self.modes(a.mode, b.mode).chain |m| {
        do self.contratys(a.ty, b.ty).chain |t| {
            Ok(arg {mode: m, ty: t})
        }
    }
}

pub fn super_vstores<C:Combine>(
    self: &C, vk: ty::terr_vstore_kind,
    a: ty::vstore, b: ty::vstore) -> cres<ty::vstore> {
    debug!("%s.super_vstores(a=%?, b=%?)", self.tag(), a, b);

    match (a, b) {
      (ty::vstore_slice(a_r), ty::vstore_slice(b_r)) => {
        do self.contraregions(a_r, b_r).chain |r| {
            Ok(ty::vstore_slice(r))
        }
      }

      _ if a == b => {
        Ok(a)
      }

      _ => {
        Err(ty::terr_vstores_differ(vk, expected_found(self, a, b)))
      }
    }
}

pub fn super_trait_stores<C:Combine>(self: &C,
                                     vk: ty::terr_vstore_kind,
                                     a: ty::TraitStore,
                                     b: ty::TraitStore)
                                  -> cres<ty::TraitStore> {
    debug!("%s.super_vstores(a=%?, b=%?)", self.tag(), a, b);

    match (a, b) {
      (ty::RegionTraitStore(a_r), ty::RegionTraitStore(b_r)) => {
        do self.contraregions(a_r, b_r).chain |r| {
            Ok(ty::RegionTraitStore(r))
        }
      }

      _ if a == b => {
        Ok(a)
      }

      _ => {
        Err(ty::terr_trait_stores_differ(vk, expected_found(self, a, b)))
      }
    }
}

pub fn super_closure_tys<C:Combine>(
    self: &C, a_f: &ty::ClosureTy, b_f: &ty::ClosureTy) -> cres<ty::ClosureTy>
{
    let p = if_ok!(self.sigils(a_f.sigil, b_f.sigil));
    let r = if_ok!(self.contraregions(a_f.region, b_f.region));
    let purity = if_ok!(self.purities(a_f.purity, b_f.purity));
    let onceness = if_ok!(self.oncenesses(a_f.onceness, b_f.onceness));
    let sig = if_ok!(self.fn_sigs(&a_f.sig, &b_f.sig));
    Ok(ty::ClosureTy {purity: purity,
                      sigil: p,
                      onceness: onceness,
                      region: r,
                      sig: sig})
}

pub fn super_abis<C:Combine>(
    self: &C, a: AbiSet, b: AbiSet) -> cres<AbiSet>
{
    if a == b {
        Ok(a)
    } else {
        Err(ty::terr_abi_mismatch(expected_found(self, a, b)))
    }
}

pub fn super_bare_fn_tys<C:Combine>(
    self: &C, a_f: &ty::BareFnTy, b_f: &ty::BareFnTy) -> cres<ty::BareFnTy>
{
    let purity = if_ok!(self.purities(a_f.purity, b_f.purity));
    let abi = if_ok!(self.abis(a_f.abis, b_f.abis));
    let sig = if_ok!(self.fn_sigs(&a_f.sig, &b_f.sig));
    Ok(ty::BareFnTy {purity: purity,
                     abis: abi,
                     sig: sig})
}

pub fn super_fn_sigs<C:Combine>(
    self: &C, a_f: &ty::FnSig, b_f: &ty::FnSig) -> cres<ty::FnSig>
{
    fn argvecs<C:Combine>(self: &C,
                          a_args: &[ty::arg],
                          b_args: &[ty::arg]) -> cres<~[ty::arg]>
    {
        if vec::same_length(a_args, b_args) {
            map_vec2(a_args, b_args, |a, b| self.args(*a, *b))
        } else {
            Err(ty::terr_arg_count)
        }
    }

    do argvecs(self, a_f.inputs, b_f.inputs)
            .chain |inputs| {
        do self.tys(a_f.output, b_f.output).chain |output| {
            Ok(FnSig {bound_lifetime_names: opt_vec::Empty, // FIXME(#4846)
                      inputs: /*bad*/copy inputs,
                      output: output})
        }
    }
}

pub fn super_tys<C:Combine>(
    self: &C, a: ty::t, b: ty::t) -> cres<ty::t> {
    let tcx = self.infcx().tcx;
    return match (/*bad*/copy ty::get(a).sty, /*bad*/copy ty::get(b).sty) {
      // The "subtype" ought to be handling cases involving bot or var:
      (ty::ty_bot, _) |
      (_, ty::ty_bot) |
      (ty::ty_infer(TyVar(_)), _) |
      (_, ty::ty_infer(TyVar(_))) => {
        tcx.sess.bug(
            fmt!("%s: bot and var types should have been handled (%s,%s)",
                 self.tag(),
                 a.inf_str(self.infcx()),
                 b.inf_str(self.infcx())));
      }

        // Relate integral variables to other types
        (ty::ty_infer(IntVar(a_id)), ty::ty_infer(IntVar(b_id))) => {
            if_ok!(self.infcx().simple_vars(self.a_is_expected(),
                                            a_id, b_id));
            Ok(a)
        }
        (ty::ty_infer(IntVar(v_id)), ty::ty_int(v)) => {
            unify_integral_variable(self, self.a_is_expected(),
                                    v_id, IntType(v))
        }
        (ty::ty_int(v), ty::ty_infer(IntVar(v_id))) => {
            unify_integral_variable(self, !self.a_is_expected(),
                                    v_id, IntType(v))
        }
        (ty::ty_infer(IntVar(v_id)), ty::ty_uint(v)) => {
            unify_integral_variable(self, self.a_is_expected(),
                                    v_id, UintType(v))
        }
        (ty::ty_uint(v), ty::ty_infer(IntVar(v_id))) => {
            unify_integral_variable(self, !self.a_is_expected(),
                                    v_id, UintType(v))
        }

        // Relate floating-point variables to other types
        (ty::ty_infer(FloatVar(a_id)), ty::ty_infer(FloatVar(b_id))) => {
            if_ok!(self.infcx().simple_vars(self.a_is_expected(),
                                            a_id, b_id));
            Ok(a)
        }
        (ty::ty_infer(FloatVar(v_id)), ty::ty_float(v)) => {
            unify_float_variable(self, self.a_is_expected(), v_id, v)
        }
        (ty::ty_float(v), ty::ty_infer(FloatVar(v_id))) => {
            unify_float_variable(self, !self.a_is_expected(), v_id, v)
        }

      (ty::ty_int(_), _) |
      (ty::ty_uint(_), _) |
      (ty::ty_float(_), _) => {
        if ty::get(a).sty == ty::get(b).sty {
            Ok(a)
        } else {
            Err(ty::terr_sorts(expected_found(self, a, b)))
        }
      }

      (ty::ty_nil, _) |
      (ty::ty_bool, _) => {
        let cfg = tcx.sess.targ_cfg;
        if ty::mach_sty(cfg, a) == ty::mach_sty(cfg, b) {
            Ok(a)
        } else {
            Err(ty::terr_sorts(expected_found(self, a, b)))
        }
      }

      (ty::ty_param(ref a_p), ty::ty_param(ref b_p)) if a_p.idx == b_p.idx => {
        Ok(a)
      }

      (ty::ty_enum(a_id, ref a_substs),
       ty::ty_enum(b_id, ref b_substs))
      if a_id == b_id => {
          let type_def = ty::lookup_item_type(tcx, a_id);
          do self.substs(&type_def.generics, a_substs, b_substs).chain |substs| {
              Ok(ty::mk_enum(tcx, a_id, substs))
          }
      }

      (ty::ty_trait(a_id, ref a_substs, a_store, a_mutbl),
       ty::ty_trait(b_id, ref b_substs, b_store, b_mutbl))
      if a_id == b_id && a_mutbl == b_mutbl => {
          let trait_def = ty::lookup_trait_def(tcx, a_id);
          do self.substs(&trait_def.generics, a_substs, b_substs).chain |substs| {
              do self.trait_stores(ty::terr_trait, a_store, b_store).chain |s| {
                  Ok(ty::mk_trait(tcx, a_id, /*bad*/copy substs, s, a_mutbl))
              }
          }
      }

      (ty::ty_struct(a_id, ref a_substs), ty::ty_struct(b_id, ref b_substs))
      if a_id == b_id => {
          let type_def = ty::lookup_item_type(tcx, a_id);
          do self.substs(&type_def.generics, a_substs, b_substs).chain |substs| {
              Ok(ty::mk_struct(tcx, a_id, substs))
          }
      }

      (ty::ty_box(ref a_mt), ty::ty_box(ref b_mt)) => {
        do self.mts(a_mt, b_mt).chain |mt| {
            Ok(ty::mk_box(tcx, mt))
        }
      }

      (ty::ty_uniq(ref a_mt), ty::ty_uniq(ref b_mt)) => {
        do self.mts(a_mt, b_mt).chain |mt| {
            Ok(ty::mk_uniq(tcx, mt))
        }
      }

      (ty::ty_ptr(ref a_mt), ty::ty_ptr(ref b_mt)) => {
        do self.mts(a_mt, b_mt).chain |mt| {
            Ok(ty::mk_ptr(tcx, mt))
        }
      }

      (ty::ty_rptr(a_r, ref a_mt), ty::ty_rptr(b_r, ref b_mt)) => {
          let r = if_ok!(self.contraregions(a_r, b_r));
          let mt = if_ok!(self.mts(a_mt, b_mt));
          Ok(ty::mk_rptr(tcx, r, mt))
      }

      (ty::ty_evec(ref a_mt, vs_a), ty::ty_evec(ref b_mt, vs_b)) => {
        do self.mts(a_mt, b_mt).chain |mt| {
            do self.vstores(ty::terr_vec, vs_a, vs_b).chain |vs| {
                Ok(ty::mk_evec(tcx, mt, vs))
            }
        }
      }

      (ty::ty_estr(vs_a), ty::ty_estr(vs_b)) => {
        do self.vstores(ty::terr_str, vs_a, vs_b).chain |vs| {
            Ok(ty::mk_estr(tcx,vs))
        }
      }

      (ty::ty_tup(ref as_), ty::ty_tup(ref bs)) => {
        if as_.len() == bs.len() {
            map_vec2(*as_, *bs, |a, b| self.tys(*a, *b) )
                .chain(|ts| Ok(ty::mk_tup(tcx, ts)) )
        } else {
            Err(ty::terr_tuple_size(
                expected_found(self, as_.len(), bs.len())))
        }
      }

      (ty::ty_bare_fn(ref a_fty), ty::ty_bare_fn(ref b_fty)) => {
        do self.bare_fn_tys(a_fty, b_fty).chain |fty| {
            Ok(ty::mk_bare_fn(tcx, fty))
        }
      }

      (ty::ty_closure(ref a_fty), ty::ty_closure(ref b_fty)) => {
        do self.closure_tys(a_fty, b_fty).chain |fty| {
            Ok(ty::mk_closure(tcx, fty))
        }
      }

      _ => Err(ty::terr_sorts(expected_found(self, a, b)))
    };

    fn unify_integral_variable<C:Combine>(
        self: &C,
        vid_is_expected: bool,
        vid: ty::IntVid,
        val: ty::IntVarValue) -> cres<ty::t>
    {
        let tcx = self.infcx().tcx;
        if val == IntType(ast::ty_char) {
            Err(ty::terr_integer_as_char)
        } else {
            if_ok!(self.infcx().simple_var_t(vid_is_expected, vid, val));
            match val {
                IntType(v) => Ok(ty::mk_mach_int(tcx, v)),
                UintType(v) => Ok(ty::mk_mach_uint(tcx, v))
            }
        }
    }

    fn unify_float_variable<C:Combine>(
        self: &C,
        vid_is_expected: bool,
        vid: ty::FloatVid,
        val: ast::float_ty) -> cres<ty::t>
    {
        let tcx = self.infcx().tcx;
        if_ok!(self.infcx().simple_var_t(vid_is_expected, vid, val));
        Ok(ty::mk_mach_float(tcx, val))
    }
}

pub fn super_trait_refs<C:Combine>(
    self: &C, a: &ty::TraitRef, b: &ty::TraitRef) -> cres<ty::TraitRef>
{
    // Different traits cannot be related

    // - NOTE in the future, expand out subtraits!

    if a.def_id != b.def_id {
        Err(ty::terr_traits(
            expected_found(self, a.def_id, b.def_id)))
    } else {
        let tcx = self.infcx().tcx;
        let trait_def = ty::lookup_trait_def(tcx, a.def_id);
        let substs = if_ok!(self.substs(&trait_def.generics, &a.substs, &b.substs));
        Ok(ty::TraitRef {
            def_id: a.def_id,
            substs: substs
        })
    }
}

