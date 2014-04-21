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
// the trait `Combine` and contains methods for combining two
// instances of various things and yielding a new instance.  These
// combiner methods always yield a `result<T>`---failure is propagated
// upward using `and_then()` methods.  There is a lot of common code for
// these operations, implemented as default methods on the `Combine`
// trait.
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


use middle::ty::{FloatVar, FnSig, IntVar, TyVar};
use middle::ty::{IntType, UintType, substs};
use middle::ty::{BuiltinBounds};
use middle::ty;
use middle::typeck::infer::{then, ToUres};
use middle::typeck::infer::glb::Glb;
use middle::typeck::infer::lub::Lub;
use middle::typeck::infer::sub::Sub;
use middle::typeck::infer::to_str::InferStr;
use middle::typeck::infer::unify::InferCtxtMethods;
use middle::typeck::infer::{InferCtxt, cres, ures};
use middle::typeck::infer::{TypeTrace};
use util::common::indent;
use util::ppaux::Repr;

use std::result;

use syntax::ast::{Onceness, FnStyle};
use syntax::ast;
use syntax::owned_slice::OwnedSlice;
use syntax::abi;

pub trait Combine {
    fn infcx<'a>(&'a self) -> &'a InferCtxt<'a>;
    fn tag(&self) -> ~str;
    fn a_is_expected(&self) -> bool;
    fn trace(&self) -> TypeTrace;

    fn sub<'a>(&'a self) -> Sub<'a>;
    fn lub<'a>(&'a self) -> Lub<'a>;
    fn glb<'a>(&'a self) -> Glb<'a>;

    fn mts(&self, a: &ty::mt, b: &ty::mt) -> cres<ty::mt>;
    fn contratys(&self, a: ty::t, b: ty::t) -> cres<ty::t>;
    fn tys(&self, a: ty::t, b: ty::t) -> cres<ty::t>;

    fn tps(&self, as_: &[ty::t], bs: &[ty::t]) -> cres<Vec<ty::t> > {

        // Note: type parameters are always treated as *invariant*
        // (otherwise the type system would be unsound).  In the
        // future we could allow type parameters to declare a
        // variance.

        if as_.len() == bs.len() {
            result::fold_(as_.iter().zip(bs.iter())
                          .map(|(a, b)| eq_tys(self, *a, *b)))
                .then(|| Ok(Vec::from_slice(as_)))
        } else {
            Err(ty::terr_ty_param_size(expected_found(self,
                                                      as_.len(),
                                                      bs.len())))
        }
    }

    fn self_tys(&self, a: Option<ty::t>, b: Option<ty::t>)
               -> cres<Option<ty::t>> {

        match (a, b) {
            (None, None) => {
                Ok(None)
            }
            (Some(a), Some(b)) => {
                // FIXME(#5781) this should be eq_tys
                // eq_tys(self, a, b).then(|| Ok(Some(a)) )
                self.contratys(a, b).and_then(|t| Ok(Some(t)))
            }
            (None, Some(_)) |
                (Some(_), None) => {
                // I think it should never happen that we unify two
                // substs and one of them has a self_ty and one
                // doesn't...? I could be wrong about this.
                self.infcx().tcx.sess.bug(
                                          format!("substitution a had a self_ty \
                                               and substitution b didn't, \
                                               or vice versa"));
            }
        }
    }

    fn substs(&self,
              item_def_id: ast::DefId,
              as_: &ty::substs,
              bs: &ty::substs) -> cres<ty::substs> {

        fn relate_region_params<C:Combine>(this: &C,
                                           item_def_id: ast::DefId,
                                           a: &ty::RegionSubsts,
                                           b: &ty::RegionSubsts)
                                           -> cres<ty::RegionSubsts> {
            let tcx = this.infcx().tcx;
            match (a, b) {
                (&ty::ErasedRegions, _) | (_, &ty::ErasedRegions) => {
                    Ok(ty::ErasedRegions)
                }

                (&ty::NonerasedRegions(ref a_rs),
                 &ty::NonerasedRegions(ref b_rs)) => {
                    let variances = ty::item_variances(tcx, item_def_id);
                    let region_params = &variances.region_params;
                    let num_region_params = region_params.len();

                    debug!("relate_region_params(\
                            item_def_id={}, \
                            a_rs={}, \
                            b_rs={},
                            region_params={})",
                            item_def_id.repr(tcx),
                            a_rs.repr(tcx),
                            b_rs.repr(tcx),
                            region_params.repr(tcx));

                    assert_eq!(num_region_params, a_rs.len());
                    assert_eq!(num_region_params, b_rs.len());
                    let mut rs = vec!();
                    for i in range(0, num_region_params) {
                        let a_r = *a_rs.get(i);
                        let b_r = *b_rs.get(i);
                        let variance = *region_params.get(i);
                        let r = match variance {
                            ty::Invariant => {
                                eq_regions(this, a_r, b_r)
                                    .and_then(|()| Ok(a_r))
                            }
                            ty::Covariant => this.regions(a_r, b_r),
                            ty::Contravariant => this.contraregions(a_r, b_r),
                            ty::Bivariant => Ok(a_r),
                        };
                        rs.push(if_ok!(r));
                    }
                    Ok(ty::NonerasedRegions(OwnedSlice::from_vec(rs)))
                }
            }
        }

        let tps = if_ok!(self.tps(as_.tps.as_slice(), bs.tps.as_slice()));
        let self_ty = if_ok!(self.self_tys(as_.self_ty, bs.self_ty));
        let regions = if_ok!(relate_region_params(self,
                                                  item_def_id,
                                                  &as_.regions,
                                                  &bs.regions));
        Ok(substs { regions: regions,
                    self_ty: self_ty,
                    tps: tps.clone() })
    }

    fn bare_fn_tys(&self, a: &ty::BareFnTy,
                   b: &ty::BareFnTy) -> cres<ty::BareFnTy> {
        let fn_style = if_ok!(self.fn_styles(a.fn_style, b.fn_style));
        let abi = if_ok!(self.abi(a.abi, b.abi));
        let sig = if_ok!(self.fn_sigs(&a.sig, &b.sig));
        Ok(ty::BareFnTy {fn_style: fn_style,
                abi: abi,
                sig: sig})
    }

    fn closure_tys(&self, a: &ty::ClosureTy,
                   b: &ty::ClosureTy) -> cres<ty::ClosureTy> {

        let store = match (a.store, b.store) {
            (ty::RegionTraitStore(a_r, a_m),
             ty::RegionTraitStore(b_r, b_m)) if a_m == b_m => {
                let r = if_ok!(self.contraregions(a_r, b_r));
                ty::RegionTraitStore(r, a_m)
            }

            _ if a.store == b.store => {
                a.store
            }

            _ => {
                return Err(ty::terr_sigil_mismatch(expected_found(self, a.store, b.store)))
            }
        };
        let fn_style = if_ok!(self.fn_styles(a.fn_style, b.fn_style));
        let onceness = if_ok!(self.oncenesses(a.onceness, b.onceness));
        let bounds = if_ok!(self.bounds(a.bounds, b.bounds));
        let sig = if_ok!(self.fn_sigs(&a.sig, &b.sig));
        Ok(ty::ClosureTy {
            fn_style: fn_style,
            onceness: onceness,
            store: store,
            bounds: bounds,
            sig: sig
        })
    }

    fn fn_sigs(&self, a: &ty::FnSig, b: &ty::FnSig) -> cres<ty::FnSig>;

    fn args(&self, a: ty::t, b: ty::t) -> cres<ty::t> {
        self.contratys(a, b).and_then(|t| Ok(t))
    }

    fn fn_styles(&self, a: FnStyle, b: FnStyle) -> cres<FnStyle>;

    fn abi(&self, a: abi::Abi, b: abi::Abi) -> cres<abi::Abi> {
        if a == b {
            Ok(a)
        } else {
            Err(ty::terr_abi_mismatch(expected_found(self, a, b)))
        }
    }

    fn oncenesses(&self, a: Onceness, b: Onceness) -> cres<Onceness>;
    fn bounds(&self, a: BuiltinBounds, b: BuiltinBounds) -> cres<BuiltinBounds>;
    fn contraregions(&self, a: ty::Region, b: ty::Region)
                  -> cres<ty::Region>;
    fn regions(&self, a: ty::Region, b: ty::Region) -> cres<ty::Region>;

    fn vstores(&self,
                vk: ty::terr_vstore_kind,
                a: ty::Vstore,
                b: ty::Vstore)
                -> cres<ty::Vstore> {
        debug!("{}.vstores(a={:?}, b={:?})", self.tag(), a, b);

        match (a, b) {
            (ty::VstoreSlice(a_r), ty::VstoreSlice(b_r)) => {
                self.contraregions(a_r, b_r).and_then(|r| {
                    Ok(ty::VstoreSlice(r))
                })
            }

            _ if a == b => {
                Ok(a)
            }

            _ => {
                Err(ty::terr_vstores_differ(vk, expected_found(self, a, b)))
            }
        }
    }

    fn trait_stores(&self,
                    vk: ty::terr_vstore_kind,
                    a: ty::TraitStore,
                    b: ty::TraitStore)
                    -> cres<ty::TraitStore> {
        debug!("{}.trait_stores(a={:?}, b={:?})", self.tag(), a, b);

        match (a, b) {
            (ty::RegionTraitStore(a_r, a_m),
             ty::RegionTraitStore(b_r, b_m)) if a_m == b_m => {
                self.contraregions(a_r, b_r).and_then(|r| {
                    Ok(ty::RegionTraitStore(r, a_m))
                })
            }

            _ if a == b => {
                Ok(a)
            }

            _ => {
                Err(ty::terr_trait_stores_differ(vk, expected_found(self, a, b)))
            }
        }

    }

    fn trait_refs(&self,
                  a: &ty::TraitRef,
                  b: &ty::TraitRef)
                  -> cres<ty::TraitRef> {
        // Different traits cannot be related

        // - NOTE in the future, expand out subtraits!

        if a.def_id != b.def_id {
            Err(ty::terr_traits(
                                expected_found(self, a.def_id, b.def_id)))
        } else {
            let substs = if_ok!(self.substs(a.def_id, &a.substs, &b.substs));
            Ok(ty::TraitRef { def_id: a.def_id,
                              substs: substs })
        }
    }
}

#[deriving(Clone)]
pub struct CombineFields<'a> {
    pub infcx: &'a InferCtxt<'a>,
    pub a_is_expected: bool,
    pub trace: TypeTrace,
}

pub fn expected_found<C:Combine,T>(
        this: &C, a: T, b: T) -> ty::expected_found<T> {
    if this.a_is_expected() {
        ty::expected_found {expected: a, found: b}
    } else {
        ty::expected_found {expected: b, found: a}
    }
}

pub fn eq_tys<C:Combine>(this: &C, a: ty::t, b: ty::t) -> ures {
    let suber = this.sub();
    this.infcx().try(|| {
        suber.tys(a, b).and_then(|_ok| suber.contratys(a, b)).to_ures()
    })
}

pub fn eq_regions<C:Combine>(this: &C, a: ty::Region, b: ty::Region)
                          -> ures {
    debug!("eq_regions({}, {})",
            a.repr(this.infcx().tcx),
            b.repr(this.infcx().tcx));
    let sub = this.sub();
    indent(|| {
        this.infcx().try(|| {
            sub.regions(a, b).and_then(|_r| sub.contraregions(a, b))
        }).or_else(|e| {
            // substitute a better error, but use the regions
            // found in the original error
            match e {
              ty::terr_regions_does_not_outlive(a1, b1) =>
                Err(ty::terr_regions_not_same(a1, b1)),
              _ => Err(e)
            }
        }).to_ures()
    })
}

pub fn super_fn_sigs<C:Combine>(this: &C, a: &ty::FnSig, b: &ty::FnSig) -> cres<ty::FnSig> {

    fn argvecs<C:Combine>(this: &C, a_args: &[ty::t], b_args: &[ty::t]) -> cres<Vec<ty::t> > {
        if a_args.len() == b_args.len() {
            result::collect(a_args.iter().zip(b_args.iter())
                            .map(|(a, b)| this.args(*a, *b)))
        } else {
            Err(ty::terr_arg_count)
        }
    }

    if a.variadic != b.variadic {
        return Err(ty::terr_variadic_mismatch(expected_found(this, a.variadic, b.variadic)));
    }

    let inputs = if_ok!(argvecs(this,
                                a.inputs.as_slice(),
                                b.inputs.as_slice()));
    let output = if_ok!(this.tys(a.output, b.output));
    Ok(FnSig {binder_id: a.binder_id,
              inputs: inputs,
              output: output,
              variadic: a.variadic})
}

pub fn super_tys<C:Combine>(this: &C, a: ty::t, b: ty::t) -> cres<ty::t> {
    let tcx = this.infcx().tcx;
    let a_sty = &ty::get(a).sty;
    let b_sty = &ty::get(b).sty;
    debug!("super_tys: a_sty={:?} b_sty={:?}", a_sty, b_sty);
    return match (a_sty, b_sty) {
      // The "subtype" ought to be handling cases involving bot or var:
      (&ty::ty_bot, _) |
      (_, &ty::ty_bot) |
      (&ty::ty_infer(TyVar(_)), _) |
      (_, &ty::ty_infer(TyVar(_))) => {
        tcx.sess.bug(
            format!("{}: bot and var types should have been handled ({},{})",
                 this.tag(),
                 a.inf_str(this.infcx()),
                 b.inf_str(this.infcx())));
      }

        // Relate integral variables to other types
        (&ty::ty_infer(IntVar(a_id)), &ty::ty_infer(IntVar(b_id))) => {
            if_ok!(this.infcx().simple_vars(this.a_is_expected(),
                                            a_id, b_id));
            Ok(a)
        }
        (&ty::ty_infer(IntVar(v_id)), &ty::ty_int(v)) => {
            unify_integral_variable(this, this.a_is_expected(),
                                    v_id, IntType(v))
        }
        (&ty::ty_int(v), &ty::ty_infer(IntVar(v_id))) => {
            unify_integral_variable(this, !this.a_is_expected(),
                                    v_id, IntType(v))
        }
        (&ty::ty_infer(IntVar(v_id)), &ty::ty_uint(v)) => {
            unify_integral_variable(this, this.a_is_expected(),
                                    v_id, UintType(v))
        }
        (&ty::ty_uint(v), &ty::ty_infer(IntVar(v_id))) => {
            unify_integral_variable(this, !this.a_is_expected(),
                                    v_id, UintType(v))
        }

        // Relate floating-point variables to other types
        (&ty::ty_infer(FloatVar(a_id)), &ty::ty_infer(FloatVar(b_id))) => {
            if_ok!(this.infcx().simple_vars(this.a_is_expected(),
                                            a_id, b_id));
            Ok(a)
        }
        (&ty::ty_infer(FloatVar(v_id)), &ty::ty_float(v)) => {
            unify_float_variable(this, this.a_is_expected(), v_id, v)
        }
        (&ty::ty_float(v), &ty::ty_infer(FloatVar(v_id))) => {
            unify_float_variable(this, !this.a_is_expected(), v_id, v)
        }

      (&ty::ty_char, _) |
      (&ty::ty_nil, _) |
      (&ty::ty_bool, _) |
      (&ty::ty_int(_), _) |
      (&ty::ty_uint(_), _) |
      (&ty::ty_float(_), _) => {
        if ty::get(a).sty == ty::get(b).sty {
            Ok(a)
        } else {
            Err(ty::terr_sorts(expected_found(this, a, b)))
        }
      }

      (&ty::ty_param(ref a_p), &ty::ty_param(ref b_p)) if a_p.idx == b_p.idx => {
        Ok(a)
      }

      (&ty::ty_enum(a_id, ref a_substs),
       &ty::ty_enum(b_id, ref b_substs))
      if a_id == b_id => {
          let substs = if_ok!(this.substs(a_id,
                                          a_substs,
                                          b_substs));
          Ok(ty::mk_enum(tcx, a_id, substs))
      }

      (&ty::ty_trait(ref a_),
       &ty::ty_trait(ref b_))
      if a_.def_id == b_.def_id => {
          debug!("Trying to match traits {:?} and {:?}", a, b);
          let substs = if_ok!(this.substs(a_.def_id, &a_.substs, &b_.substs));
          let s = if_ok!(this.trait_stores(ty::terr_trait, a_.store, b_.store));
          let bounds = if_ok!(this.bounds(a_.bounds, b_.bounds));
          Ok(ty::mk_trait(tcx,
                          a_.def_id,
                          substs.clone(),
                          s,
                          bounds))
      }

      (&ty::ty_struct(a_id, ref a_substs), &ty::ty_struct(b_id, ref b_substs))
      if a_id == b_id => {
            let substs = if_ok!(this.substs(a_id, a_substs, b_substs));
            Ok(ty::mk_struct(tcx, a_id, substs))
      }

      (&ty::ty_box(a_inner), &ty::ty_box(b_inner)) => {
        this.tys(a_inner, b_inner).and_then(|typ| Ok(ty::mk_box(tcx, typ)))
      }

      (&ty::ty_uniq(a_inner), &ty::ty_uniq(b_inner)) => {
            let typ = if_ok!(this.tys(a_inner, b_inner));

            match (&ty::get(a_inner).sty, &ty::get(b_inner).sty) {
                (&ty::ty_vec(_, None), &ty::ty_vec(_, None)) => Ok(ty::mk_uniq(tcx, typ)),
                (&ty::ty_vec(_, None), _) |
                (_, &ty::ty_vec(_, None)) => Err(ty::terr_sorts(expected_found(this, a, b))),
                _ => Ok(ty::mk_uniq(tcx, typ)),
            }
      }

      (&ty::ty_ptr(ref a_mt), &ty::ty_ptr(ref b_mt)) => {
            let mt = if_ok!(this.mts(a_mt, b_mt));
            match (&ty::get(a_mt.ty).sty, &ty::get(b_mt.ty).sty) {
                (&ty::ty_vec(_, None), &ty::ty_vec(_, None)) => Ok(ty::mk_ptr(tcx, mt)),
                (&ty::ty_vec(_, None), _) |
                (_, &ty::ty_vec(_, None)) => Err(ty::terr_sorts(expected_found(this, a, b))),
                _ => Ok(ty::mk_ptr(tcx, mt)),
            }
      }

      (&ty::ty_rptr(a_r, ref a_mt), &ty::ty_rptr(b_r, ref b_mt)) => {
            let r = if_ok!(this.contraregions(a_r, b_r));
            let mt = if_ok!(this.mts(a_mt, b_mt));

            // This is a horible hack - historically, [T] was not treated as a type,
            // so, for example, &T and &[U] should not unify. In fact the only thing
            // &[U] should unify with is &[T]. We preserve that behaviour with this
            // check. See also ty_uniq, ty_ptr.
            match (&ty::get(a_mt.ty).sty, &ty::get(b_mt.ty).sty) {
                (&ty::ty_vec(_, None), &ty::ty_vec(_, None)) => Ok(ty::mk_rptr(tcx, r, mt)),
                (&ty::ty_vec(_, None), _) |
                (_, &ty::ty_vec(_, None)) => Err(ty::terr_sorts(expected_found(this, a, b))),
                _ => Ok(ty::mk_rptr(tcx, r, mt)),
            }
      }

      (&ty::ty_vec(ref a_mt, sz_a), &ty::ty_vec(ref b_mt, sz_b)) => {
        this.mts(a_mt, b_mt).and_then(|mt| {
            if sz_a == sz_b {
                Ok(ty::mk_vec(tcx, mt, sz_a))
            } else {
                Err(ty::terr_sorts(expected_found(this, a, b)))
            }
        })
      }

      (&ty::ty_str(vs_a), &ty::ty_str(vs_b)) => {
        let vs = if_ok!(this.vstores(ty::terr_str, vs_a, vs_b));
        Ok(ty::mk_str(tcx,vs))
      }

      (&ty::ty_tup(ref as_), &ty::ty_tup(ref bs)) => {
        if as_.len() == bs.len() {
            result::collect(as_.iter().zip(bs.iter())
                            .map(|(a, b)| this.tys(*a, *b)))
                    .and_then(|ts| Ok(ty::mk_tup(tcx, ts)) )
        } else {
            Err(ty::terr_tuple_size(
                expected_found(this, as_.len(), bs.len())))
        }
      }

      (&ty::ty_bare_fn(ref a_fty), &ty::ty_bare_fn(ref b_fty)) => {
        this.bare_fn_tys(a_fty, b_fty).and_then(|fty| {
            Ok(ty::mk_bare_fn(tcx, fty))
        })
      }

      (&ty::ty_closure(ref a_fty), &ty::ty_closure(ref b_fty)) => {
        this.closure_tys(*a_fty, *b_fty).and_then(|fty| {
            Ok(ty::mk_closure(tcx, fty))
        })
      }

      _ => Err(ty::terr_sorts(expected_found(this, a, b)))
    };

    fn unify_integral_variable<C:Combine>(
        this: &C,
        vid_is_expected: bool,
        vid: ty::IntVid,
        val: ty::IntVarValue) -> cres<ty::t>
    {
        if_ok!(this.infcx().simple_var_t(vid_is_expected, vid, val));
        match val {
            IntType(v) => Ok(ty::mk_mach_int(v)),
            UintType(v) => Ok(ty::mk_mach_uint(v))
        }
    }

    fn unify_float_variable<C:Combine>(
        this: &C,
        vid_is_expected: bool,
        vid: ty::FloatVid,
        val: ast::FloatTy) -> cres<ty::t>
    {
        if_ok!(this.infcx().simple_var_t(vid_is_expected, vid, val));
        Ok(ty::mk_mach_float(val))
    }
}
