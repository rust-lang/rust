// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

///////////////////////////////////////////////////////////////////////////
// # Type combining
//
// There are four type combiners: equate, sub, lub, and glb.  Each
// implements the trait `Combine` and contains methods for combining
// two instances of various things and yielding a new instance.  These
// combiner methods always yield a `Result<T>`.  There is a lot of
// common code for these operations, implemented as default methods on
// the `Combine` trait.
//
// Each operation may have side-effects on the inference context,
// though these can be unrolled using snapshots. On success, the
// LUB/GLB operations return the appropriate bound. The Eq and Sub
// operations generally return the first operand.
//
// ## Contravariance
//
// When you are relating two things which have a contravariant
// relationship, you should use `contratys()` or `contraregions()`,
// rather than inversing the order of arguments!  This is necessary
// because the order of arguments is not relevant for LUB and GLB.  It
// is also useful to track which value is the "expected" value in
// terms of error reporting.


use middle::subst;
use middle::subst::{ErasedRegions, NonerasedRegions, Substs};
use middle::ty::{FloatVar, FnSig, IntVar, TyVar};
use middle::ty::{IntType, UintType};
use middle::ty::{BuiltinBounds};
use middle::ty;
use middle::ty_fold;
use middle::typeck::infer::equate::Equate;
use middle::typeck::infer::glb::Glb;
use middle::typeck::infer::lub::Lub;
use middle::typeck::infer::sub::Sub;
use middle::typeck::infer::unify::InferCtxtMethodsForSimplyUnifiableTypes;
use middle::typeck::infer::{InferCtxt, cres};
use middle::typeck::infer::{MiscVariable, TypeTrace};
use middle::typeck::infer::type_variable::{RelationDir, EqTo,
                                           SubtypeOf, SupertypeOf};
use middle::ty_fold::{TypeFoldable};
use util::ppaux::Repr;

use std::result;

use syntax::ast::{Onceness, FnStyle};
use syntax::ast;
use syntax::abi;
use syntax::codemap::Span;

pub trait Combine<'tcx> {
    fn infcx<'a>(&'a self) -> &'a InferCtxt<'a, 'tcx>;
    fn tag(&self) -> String;
    fn a_is_expected(&self) -> bool;
    fn trace(&self) -> TypeTrace;

    fn equate<'a>(&'a self) -> Equate<'a, 'tcx>;
    fn sub<'a>(&'a self) -> Sub<'a, 'tcx>;
    fn lub<'a>(&'a self) -> Lub<'a, 'tcx>;
    fn glb<'a>(&'a self) -> Glb<'a, 'tcx>;

    fn mts(&self, a: &ty::mt, b: &ty::mt) -> cres<ty::mt>;
    fn contratys(&self, a: ty::t, b: ty::t) -> cres<ty::t>;
    fn tys(&self, a: ty::t, b: ty::t) -> cres<ty::t>;

    fn tps(&self,
           _: subst::ParamSpace,
           as_: &[ty::t],
           bs: &[ty::t])
           -> cres<Vec<ty::t>> {
        // FIXME -- In general, we treat variance a bit wrong
        // here. For historical reasons, we treat tps and Self
        // as invariant. This is overly conservative.

        if as_.len() != bs.len() {
            return Err(ty::terr_ty_param_size(expected_found(self,
                                                             as_.len(),
                                                             bs.len())));
        }

        try!(result::fold_(as_
                          .iter()
                          .zip(bs.iter())
                          .map(|(a, b)| self.equate().tys(*a, *b))));
        Ok(Vec::from_slice(as_))
    }

    fn substs(&self,
              item_def_id: ast::DefId,
              a_subst: &subst::Substs,
              b_subst: &subst::Substs)
              -> cres<subst::Substs>
    {
        let variances = if self.infcx().tcx.variance_computed.get() {
            Some(ty::item_variances(self.infcx().tcx, item_def_id))
        } else {
            None
        };
        let mut substs = subst::Substs::empty();

        for &space in subst::ParamSpace::all().iter() {
            let a_tps = a_subst.types.get_slice(space);
            let b_tps = b_subst.types.get_slice(space);
            let tps = try!(self.tps(space, a_tps, b_tps));
            substs.types.replace(space, tps);
        }

        match (&a_subst.regions, &b_subst.regions) {
            (&ErasedRegions, _) | (_, &ErasedRegions) => {
                substs.regions = ErasedRegions;
            }

            (&NonerasedRegions(ref a), &NonerasedRegions(ref b)) => {
                for &space in subst::ParamSpace::all().iter() {
                    let a_regions = a.get_slice(space);
                    let b_regions = b.get_slice(space);

                    let mut invariance = Vec::new();
                    let r_variances = match variances {
                        Some(ref variances) => {
                            variances.regions.get_slice(space)
                        }
                        None => {
                            for _ in a_regions.iter() {
                                invariance.push(ty::Invariant);
                            }
                            invariance.as_slice()
                        }
                    };

                    let regions = try!(relate_region_params(self,
                                                            item_def_id,
                                                            r_variances,
                                                            a_regions,
                                                            b_regions));
                    substs.mut_regions().replace(space, regions);
                }
            }
        }

        return Ok(substs);

        fn relate_region_params<'tcx, C: Combine<'tcx>>(this: &C,
                                                        item_def_id: ast::DefId,
                                                        variances: &[ty::Variance],
                                                        a_rs: &[ty::Region],
                                                        b_rs: &[ty::Region])
                                                        -> cres<Vec<ty::Region>> {
            let tcx = this.infcx().tcx;
            let num_region_params = variances.len();

            debug!("relate_region_params(\
                   item_def_id={}, \
                   a_rs={}, \
                   b_rs={},
                   variances={})",
                   item_def_id.repr(tcx),
                   a_rs.repr(tcx),
                   b_rs.repr(tcx),
                   variances.repr(tcx));

            assert_eq!(num_region_params, a_rs.len());
            assert_eq!(num_region_params, b_rs.len());
            let mut rs = vec!();
            for i in range(0, num_region_params) {
                let a_r = a_rs[i];
                let b_r = b_rs[i];
                let variance = variances[i];
                let r = match variance {
                    ty::Invariant => this.equate().regions(a_r, b_r),
                    ty::Covariant => this.regions(a_r, b_r),
                    ty::Contravariant => this.contraregions(a_r, b_r),
                    ty::Bivariant => Ok(a_r),
                };
                rs.push(try!(r));
            }
            Ok(rs)
        }
    }

    fn bare_fn_tys(&self, a: &ty::BareFnTy,
                   b: &ty::BareFnTy) -> cres<ty::BareFnTy> {
        let fn_style = try!(self.fn_styles(a.fn_style, b.fn_style));
        let abi = try!(self.abi(a.abi, b.abi));
        let sig = try!(self.fn_sigs(&a.sig, &b.sig));
        Ok(ty::BareFnTy {fn_style: fn_style,
                abi: abi,
                sig: sig})
    }

    fn closure_tys(&self, a: &ty::ClosureTy,
                   b: &ty::ClosureTy) -> cres<ty::ClosureTy> {

        let store = match (a.store, b.store) {
            (ty::RegionTraitStore(a_r, a_m),
             ty::RegionTraitStore(b_r, b_m)) if a_m == b_m => {
                let r = try!(self.contraregions(a_r, b_r));
                ty::RegionTraitStore(r, a_m)
            }

            _ if a.store == b.store => {
                a.store
            }

            _ => {
                return Err(ty::terr_sigil_mismatch(expected_found(self, a.store, b.store)))
            }
        };
        let fn_style = try!(self.fn_styles(a.fn_style, b.fn_style));
        let onceness = try!(self.oncenesses(a.onceness, b.onceness));
        let bounds = try!(self.existential_bounds(a.bounds, b.bounds));
        let sig = try!(self.fn_sigs(&a.sig, &b.sig));
        let abi = try!(self.abi(a.abi, b.abi));
        Ok(ty::ClosureTy {
            fn_style: fn_style,
            onceness: onceness,
            store: store,
            bounds: bounds,
            sig: sig,
            abi: abi,
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

    fn existential_bounds(&self,
                          a: ty::ExistentialBounds,
                          b: ty::ExistentialBounds)
                          -> cres<ty::ExistentialBounds>
    {
        let r = try!(self.contraregions(a.region_bound, b.region_bound));
        let nb = try!(self.builtin_bounds(a.builtin_bounds, b.builtin_bounds));
        Ok(ty::ExistentialBounds { region_bound: r,
                                   builtin_bounds: nb })
    }

    fn builtin_bounds(&self,
                      a: ty::BuiltinBounds,
                      b: ty::BuiltinBounds)
                      -> cres<ty::BuiltinBounds>;

    fn contraregions(&self, a: ty::Region, b: ty::Region)
                  -> cres<ty::Region>;

    fn regions(&self, a: ty::Region, b: ty::Region) -> cres<ty::Region>;

    fn trait_stores(&self,
                    vk: ty::terr_vstore_kind,
                    a: ty::TraitStore,
                    b: ty::TraitStore)
                    -> cres<ty::TraitStore> {
        debug!("{}.trait_stores(a={}, b={})", self.tag(), a, b);

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
            let substs = try!(self.substs(a.def_id, &a.substs, &b.substs));
            Ok(ty::TraitRef { def_id: a.def_id,
                              substs: substs })
        }
    }
}

#[deriving(Clone)]
pub struct CombineFields<'a, 'tcx: 'a> {
    pub infcx: &'a InferCtxt<'a, 'tcx>,
    pub a_is_expected: bool,
    pub trace: TypeTrace,
}

pub fn expected_found<'tcx, C: Combine<'tcx>, T>(
        this: &C, a: T, b: T) -> ty::expected_found<T> {
    if this.a_is_expected() {
        ty::expected_found {expected: a, found: b}
    } else {
        ty::expected_found {expected: b, found: a}
    }
}

pub fn super_fn_sigs<'tcx, C: Combine<'tcx>>(this: &C,
                                             a: &ty::FnSig,
                                             b: &ty::FnSig)
                                             -> cres<ty::FnSig> {

    fn argvecs<'tcx, C: Combine<'tcx>>(this: &C,
                                       a_args: &[ty::t],
                                       b_args: &[ty::t])
                                       -> cres<Vec<ty::t>> {
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

    let inputs = try!(argvecs(this,
                                a.inputs.as_slice(),
                                b.inputs.as_slice()));
    let output = try!(this.tys(a.output, b.output));
    Ok(FnSig {binder_id: a.binder_id,
              inputs: inputs,
              output: output,
              variadic: a.variadic})
}

pub fn super_tys<'tcx, C: Combine<'tcx>>(this: &C, a: ty::t, b: ty::t) -> cres<ty::t> {

    // This is a horrible hack - historically, [T] was not treated as a type,
    // so, for example, &T and &[U] should not unify. In fact the only thing
    // &[U] should unify with is &[T]. We preserve that behaviour with this
    // check.
    fn check_ptr_to_unsized<'tcx, C: Combine<'tcx>>(this: &C,
                                                    a: ty::t,
                                                    b: ty::t,
                                                    a_inner: ty::t,
                                                    b_inner: ty::t,
                                                    result: ty::t) -> cres<ty::t> {
        match (&ty::get(a_inner).sty, &ty::get(b_inner).sty) {
            (&ty::ty_vec(_, None), &ty::ty_vec(_, None)) |
            (&ty::ty_str, &ty::ty_str) |
            (&ty::ty_trait(..), &ty::ty_trait(..)) => Ok(result),
            (&ty::ty_vec(_, None), _) | (_, &ty::ty_vec(_, None)) |
            (&ty::ty_str, _) | (_, &ty::ty_str) |
            (&ty::ty_trait(..), _) | (_, &ty::ty_trait(..))
                => Err(ty::terr_sorts(expected_found(this, a, b))),
            _ => Ok(result),
        }
    }

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
                    a.repr(this.infcx().tcx),
                    b.repr(this.infcx().tcx)).as_slice());
      }

        // Relate integral variables to other types
        (&ty::ty_infer(IntVar(a_id)), &ty::ty_infer(IntVar(b_id))) => {
            try!(this.infcx().simple_vars(this.a_is_expected(),
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
            try!(this.infcx().simple_vars(this.a_is_expected(), a_id, b_id));
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
      (&ty::ty_float(_), _) |
      (&ty::ty_err, _) => {
        if ty::get(a).sty == ty::get(b).sty {
            Ok(a)
        } else {
            Err(ty::terr_sorts(expected_found(this, a, b)))
        }
      }

      (&ty::ty_param(ref a_p), &ty::ty_param(ref b_p)) if
          a_p.idx == b_p.idx && a_p.space == b_p.space => {
        Ok(a)
      }

      (&ty::ty_enum(a_id, ref a_substs),
       &ty::ty_enum(b_id, ref b_substs))
      if a_id == b_id => {
          let substs = try!(this.substs(a_id,
                                          a_substs,
                                          b_substs));
          Ok(ty::mk_enum(tcx, a_id, substs))
      }

      (&ty::ty_trait(ref a_),
       &ty::ty_trait(ref b_))
      if a_.def_id == b_.def_id => {
          debug!("Trying to match traits {:?} and {:?}", a, b);
          let substs = try!(this.substs(a_.def_id, &a_.substs, &b_.substs));
          let bounds = try!(this.existential_bounds(a_.bounds, b_.bounds));
          Ok(ty::mk_trait(tcx,
                          a_.def_id,
                          substs.clone(),
                          bounds))
      }

      (&ty::ty_struct(a_id, ref a_substs), &ty::ty_struct(b_id, ref b_substs))
      if a_id == b_id => {
            let substs = try!(this.substs(a_id, a_substs, b_substs));
            Ok(ty::mk_struct(tcx, a_id, substs))
      }

      (&ty::ty_unboxed_closure(a_id, a_region),
       &ty::ty_unboxed_closure(b_id, b_region))
      if a_id == b_id => {
          // All ty_unboxed_closure types with the same id represent
          // the (anonymous) type of the same closure expression. So
          // all of their regions should be equated.
          let region = try!(this.equate().regions(a_region, b_region));
          Ok(ty::mk_unboxed_closure(tcx, a_id, region))
      }

      (&ty::ty_box(a_inner), &ty::ty_box(b_inner)) => {
        this.tys(a_inner, b_inner).and_then(|typ| Ok(ty::mk_box(tcx, typ)))
      }

      (&ty::ty_uniq(a_inner), &ty::ty_uniq(b_inner)) => {
            let typ = try!(this.tys(a_inner, b_inner));
            check_ptr_to_unsized(this, a, b, a_inner, b_inner, ty::mk_uniq(tcx, typ))
      }

      (&ty::ty_ptr(ref a_mt), &ty::ty_ptr(ref b_mt)) => {
            let mt = try!(this.mts(a_mt, b_mt));
            check_ptr_to_unsized(this, a, b, a_mt.ty, b_mt.ty, ty::mk_ptr(tcx, mt))
      }

      (&ty::ty_rptr(a_r, ref a_mt), &ty::ty_rptr(b_r, ref b_mt)) => {
            let r = try!(this.contraregions(a_r, b_r));
            // FIXME(14985)  If we have mutable references to trait objects, we
            // used to use covariant subtyping. I have preserved this behaviour,
            // even though it is probably incorrect. So don't go down the usual
            // path which would require invariance.
            let mt = match (&ty::get(a_mt.ty).sty, &ty::get(b_mt.ty).sty) {
                (&ty::ty_trait(..), &ty::ty_trait(..)) if a_mt.mutbl == b_mt.mutbl => {
                    let ty = try!(this.tys(a_mt.ty, b_mt.ty));
                    ty::mt { ty: ty, mutbl: a_mt.mutbl }
                }
                _ => try!(this.mts(a_mt, b_mt))
            };
            check_ptr_to_unsized(this, a, b, a_mt.ty, b_mt.ty, ty::mk_rptr(tcx, r, mt))
      }

      (&ty::ty_vec(a_t, sz_a), &ty::ty_vec(b_t, sz_b)) => {
        this.tys(a_t, b_t).and_then(|t| {
            if sz_a == sz_b {
                Ok(ty::mk_vec(tcx, t, sz_a))
            } else {
                Err(ty::terr_sorts(expected_found(this, a, b)))
            }
        })
      }

      (&ty::ty_str, &ty::ty_str) => {
            Ok(ty::mk_str(tcx))
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
        this.closure_tys(&**a_fty, &**b_fty).and_then(|fty| {
            Ok(ty::mk_closure(tcx, fty))
        })
      }

      _ => Err(ty::terr_sorts(expected_found(this, a, b)))
    };

    fn unify_integral_variable<'tcx, C: Combine<'tcx>>(
        this: &C,
        vid_is_expected: bool,
        vid: ty::IntVid,
        val: ty::IntVarValue) -> cres<ty::t>
    {
        try!(this.infcx().simple_var_t(vid_is_expected, vid, val));
        match val {
            IntType(v) => Ok(ty::mk_mach_int(v)),
            UintType(v) => Ok(ty::mk_mach_uint(v))
        }
    }

    fn unify_float_variable<'tcx, C: Combine<'tcx>>(
        this: &C,
        vid_is_expected: bool,
        vid: ty::FloatVid,
        val: ast::FloatTy) -> cres<ty::t>
    {
        try!(this.infcx().simple_var_t(vid_is_expected, vid, val));
        Ok(ty::mk_mach_float(val))
    }
}

impl<'f, 'tcx> CombineFields<'f, 'tcx> {
    pub fn switch_expected(&self) -> CombineFields<'f, 'tcx> {
        CombineFields {
            a_is_expected: !self.a_is_expected,
            ..(*self).clone()
        }
    }

    fn equate(&self) -> Equate<'f, 'tcx> {
        Equate((*self).clone())
    }

    fn sub(&self) -> Sub<'f, 'tcx> {
        Sub((*self).clone())
    }

    pub fn instantiate(&self,
                       a_ty: ty::t,
                       dir: RelationDir,
                       b_vid: ty::TyVid)
                       -> cres<()>
    {
        let tcx = self.infcx.tcx;
        let mut stack = Vec::new();
        stack.push((a_ty, dir, b_vid));
        loop {
            // For each turn of the loop, we extract a tuple
            //
            //     (a_ty, dir, b_vid)
            //
            // to relate. Here dir is either SubtypeOf or
            // SupertypeOf. The idea is that we should ensure that
            // the type `a_ty` is a subtype or supertype (respectively) of the
            // type to which `b_vid` is bound.
            //
            // If `b_vid` has not yet been instantiated with a type
            // (which is always true on the first iteration, but not
            // necessarily true on later iterations), we will first
            // instantiate `b_vid` with a *generalized* version of
            // `a_ty`. Generalization introduces other inference
            // variables wherever subtyping could occur (at time of
            // this writing, this means replacing free regions with
            // region variables).
            let (a_ty, dir, b_vid) = match stack.pop() {
                None => break,
                Some(e) => e,
            };

            debug!("instantiate(a_ty={} dir={} b_vid={})",
                   a_ty.repr(tcx),
                   dir,
                   b_vid.repr(tcx));

            // Check whether `vid` has been instantiated yet.  If not,
            // make a generalized form of `ty` and instantiate with
            // that.
            let b_ty = self.infcx.type_variables.borrow().probe(b_vid);
            let b_ty = match b_ty {
                Some(t) => t, // ...already instantiated.
                None => {     // ...not yet instantiated:
                    // Generalize type if necessary.
                    let generalized_ty = try!(match dir {
                        EqTo => {
                            self.generalize(a_ty, b_vid, false)
                        }
                        SupertypeOf | SubtypeOf => {
                            self.generalize(a_ty, b_vid, true)
                        }
                    });
                    debug!("instantiate(a_ty={}, dir={}, \
                                        b_vid={}, generalized_ty={})",
                           a_ty.repr(tcx), dir, b_vid.repr(tcx),
                           generalized_ty.repr(tcx));
                    self.infcx.type_variables
                        .borrow_mut()
                        .instantiate_and_push(
                            b_vid, generalized_ty, &mut stack);
                    generalized_ty
                }
            };

            // The original triple was `(a_ty, dir, b_vid)` -- now we have
            // resolved `b_vid` to `b_ty`, so apply `(a_ty, dir, b_ty)`:
            //
            // FIXME(#16847): This code is non-ideal because all these subtype
            // relations wind up attributed to the same spans. We need
            // to associate causes/spans with each of the relations in
            // the stack to get this right.
            match dir {
                EqTo => {
                    try!(self.equate().tys(a_ty, b_ty));
                }

                SubtypeOf => {
                    try!(self.sub().tys(a_ty, b_ty));
                }

                SupertypeOf => {
                    try!(self.sub().contratys(a_ty, b_ty));
                }
            }
        }

        Ok(())
    }

    fn generalize(&self,
                  ty: ty::t,
                  for_vid: ty::TyVid,
                  make_region_vars: bool)
                  -> cres<ty::t>
    {
        /*!
         * Attempts to generalize `ty` for the type variable
         * `for_vid`.  This checks for cycle -- that is, whether the
         * type `ty` references `for_vid`. If `make_region_vars` is
         * true, it will also replace all regions with fresh
         * variables. Returns `ty_err` in the case of a cycle, `Ok`
         * otherwise.
         */

        let mut generalize = Generalizer { infcx: self.infcx,
                                           span: self.trace.origin.span(),
                                           for_vid: for_vid,
                                           make_region_vars: make_region_vars,
                                           cycle_detected: false };
        let u = ty.fold_with(&mut generalize);
        if generalize.cycle_detected {
            Err(ty::terr_cyclic_ty)
        } else {
            Ok(u)
        }
    }
}

struct Generalizer<'cx, 'tcx:'cx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,
    span: Span,
    for_vid: ty::TyVid,
    make_region_vars: bool,
    cycle_detected: bool,
}

impl<'cx, 'tcx> ty_fold::TypeFolder<'tcx> for Generalizer<'cx, 'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: ty::t) -> ty::t {
        // Check to see whether the type we are genealizing references
        // `vid`. At the same time, also update any type variables to
        // the values that they are bound to. This is needed to truly
        // check for cycles, but also just makes things readable.
        //
        // (In particular, you could have something like `$0 = Box<$1>`
        //  where `$1` has already been instantiated with `Box<$0>`)
        match ty::get(t).sty {
            ty::ty_infer(ty::TyVar(vid)) => {
                if vid == self.for_vid {
                    self.cycle_detected = true;
                    ty::mk_err()
                } else {
                    match self.infcx.type_variables.borrow().probe(vid) {
                        Some(u) => self.fold_ty(u),
                        None => t,
                    }
                }
            }
            _ => {
                ty_fold::super_fold_ty(self, t)
            }
        }
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        match r {
            ty::ReLateBound(..) | ty::ReEarlyBound(..) => r,
            _ if self.make_region_vars => {
                // FIXME: This is non-ideal because we don't give a
                // very descriptive origin for this region variable.
                self.infcx.next_region_var(MiscVariable(self.span))
            }
            _ => r,
        }
    }
}


