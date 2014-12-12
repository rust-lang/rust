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

use super::equate::Equate;
use super::glb::Glb;
use super::lub::Lub;
use super::sub::Sub;
use super::unify::InferCtxtMethodsForSimplyUnifiableTypes;
use super::{InferCtxt, cres};
use super::{MiscVariable, TypeTrace};
use super::type_variable::{RelationDir, EqTo, SubtypeOf, SupertypeOf};

use middle::subst;
use middle::subst::{ErasedRegions, NonerasedRegions, Substs};
use middle::ty::{FloatVar, FnSig, IntVar, TyVar};
use middle::ty::{IntType, UintType};
use middle::ty::{BuiltinBounds};
use middle::ty::{mod, Ty};
use middle::ty_fold;
use middle::ty_fold::{TypeFoldable};
use util::ppaux::Repr;

use syntax::ast::{Onceness, Unsafety};
use syntax::ast;
use syntax::abi;
use syntax::codemap::Span;

pub trait Combine<'tcx> {
    fn infcx<'a>(&'a self) -> &'a InferCtxt<'a, 'tcx>;
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> { self.infcx().tcx }
    fn tag(&self) -> String;
    fn a_is_expected(&self) -> bool;
    fn trace(&self) -> TypeTrace<'tcx>;

    fn equate<'a>(&'a self) -> Equate<'a, 'tcx>;
    fn sub<'a>(&'a self) -> Sub<'a, 'tcx>;
    fn lub<'a>(&'a self) -> Lub<'a, 'tcx>;
    fn glb<'a>(&'a self) -> Glb<'a, 'tcx>;

    fn mts(&self, a: &ty::mt<'tcx>, b: &ty::mt<'tcx>) -> cres<'tcx, ty::mt<'tcx>>;
    fn contratys(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> cres<'tcx, Ty<'tcx>>;
    fn tys(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> cres<'tcx, Ty<'tcx>>;

    fn tps(&self,
           _: subst::ParamSpace,
           as_: &[Ty<'tcx>],
           bs: &[Ty<'tcx>])
           -> cres<'tcx, Vec<Ty<'tcx>>> {
        // FIXME -- In general, we treat variance a bit wrong
        // here. For historical reasons, we treat tps and Self
        // as invariant. This is overly conservative.

        if as_.len() != bs.len() {
            return Err(ty::terr_ty_param_size(expected_found(self,
                                                             as_.len(),
                                                             bs.len())));
        }

        try!(as_.iter().zip(bs.iter())
                .map(|(a, b)| self.equate().tys(*a, *b))
                .collect::<cres<Vec<Ty>>>());
        Ok(as_.to_vec())
    }

    fn substs(&self,
              item_def_id: ast::DefId,
              a_subst: &subst::Substs<'tcx>,
              b_subst: &subst::Substs<'tcx>)
              -> cres<'tcx, subst::Substs<'tcx>>
    {
        let variances = if self.infcx().tcx.variance_computed.get() {
            Some(ty::item_variances(self.infcx().tcx, item_def_id))
        } else {
            None
        };
        self.substs_variances(variances.as_ref().map(|v| &**v), a_subst, b_subst)
    }

    fn substs_variances(&self,
                        variances: Option<&ty::ItemVariances>,
                        a_subst: &subst::Substs<'tcx>,
                        b_subst: &subst::Substs<'tcx>)
                        -> cres<'tcx, subst::Substs<'tcx>>
    {
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
                        Some(variances) => {
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
                                                            r_variances,
                                                            a_regions,
                                                            b_regions));
                    substs.mut_regions().replace(space, regions);
                }
            }
        }

        return Ok(substs);

        fn relate_region_params<'tcx, C: Combine<'tcx>>(this: &C,
                                                        variances: &[ty::Variance],
                                                        a_rs: &[ty::Region],
                                                        b_rs: &[ty::Region])
                                                        -> cres<'tcx, Vec<ty::Region>> {
            let tcx = this.infcx().tcx;
            let num_region_params = variances.len();

            debug!("relate_region_params(\
                   a_rs={}, \
                   b_rs={},
                   variances={})",
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

    fn bare_fn_tys(&self, a: &ty::BareFnTy<'tcx>,
                   b: &ty::BareFnTy<'tcx>) -> cres<'tcx, ty::BareFnTy<'tcx>> {
        let unsafety = try!(self.unsafeties(a.unsafety, b.unsafety));
        let abi = try!(self.abi(a.abi, b.abi));
        let sig = try!(self.binders(&a.sig, &b.sig));
        Ok(ty::BareFnTy {unsafety: unsafety,
                         abi: abi,
                         sig: sig})
    }

    fn closure_tys(&self, a: &ty::ClosureTy<'tcx>,
                   b: &ty::ClosureTy<'tcx>) -> cres<'tcx, ty::ClosureTy<'tcx>> {

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
        let unsafety = try!(self.unsafeties(a.unsafety, b.unsafety));
        let onceness = try!(self.oncenesses(a.onceness, b.onceness));
        let bounds = try!(self.existential_bounds(a.bounds, b.bounds));
        let sig = try!(self.binders(&a.sig, &b.sig));
        let abi = try!(self.abi(a.abi, b.abi));
        Ok(ty::ClosureTy {
            unsafety: unsafety,
            onceness: onceness,
            store: store,
            bounds: bounds,
            sig: sig,
            abi: abi,
        })
    }

    fn fn_sigs(&self, a: &ty::FnSig<'tcx>, b: &ty::FnSig<'tcx>) -> cres<'tcx, ty::FnSig<'tcx>> {
        if a.variadic != b.variadic {
            return Err(ty::terr_variadic_mismatch(expected_found(self, a.variadic, b.variadic)));
        }

        let inputs = try!(argvecs(self,
                                  a.inputs.as_slice(),
                                  b.inputs.as_slice()));

        let output = try!(match (a.output, b.output) {
            (ty::FnConverging(a_ty), ty::FnConverging(b_ty)) =>
                Ok(ty::FnConverging(try!(self.tys(a_ty, b_ty)))),
            (ty::FnDiverging, ty::FnDiverging) =>
                Ok(ty::FnDiverging),
            (a, b) =>
                Err(ty::terr_convergence_mismatch(
                    expected_found(self, a != ty::FnDiverging, b != ty::FnDiverging))),
        });

        return Ok(ty::FnSig {inputs: inputs,
                             output: output,
                             variadic: a.variadic});


        fn argvecs<'tcx, C: Combine<'tcx>>(combiner: &C,
                                           a_args: &[Ty<'tcx>],
                                           b_args: &[Ty<'tcx>])
                                           -> cres<'tcx, Vec<Ty<'tcx>>>
        {
            if a_args.len() == b_args.len() {
                a_args.iter().zip(b_args.iter())
                    .map(|(a, b)| combiner.args(*a, *b)).collect()
            } else {
                Err(ty::terr_arg_count)
            }
        }
    }

    fn args(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> cres<'tcx, Ty<'tcx>> {
        self.contratys(a, b).and_then(|t| Ok(t))
    }

    fn unsafeties(&self, a: Unsafety, b: Unsafety) -> cres<'tcx, Unsafety>;

    fn abi(&self, a: abi::Abi, b: abi::Abi) -> cres<'tcx, abi::Abi> {
        if a == b {
            Ok(a)
        } else {
            Err(ty::terr_abi_mismatch(expected_found(self, a, b)))
        }
    }

    fn oncenesses(&self, a: Onceness, b: Onceness) -> cres<'tcx, Onceness>;

    fn existential_bounds(&self,
                          a: ty::ExistentialBounds,
                          b: ty::ExistentialBounds)
                          -> cres<'tcx, ty::ExistentialBounds>
    {
        let r = try!(self.contraregions(a.region_bound, b.region_bound));
        let nb = try!(self.builtin_bounds(a.builtin_bounds, b.builtin_bounds));
        Ok(ty::ExistentialBounds { region_bound: r,
                                   builtin_bounds: nb })
    }

    fn builtin_bounds(&self,
                      a: ty::BuiltinBounds,
                      b: ty::BuiltinBounds)
                      -> cres<'tcx, ty::BuiltinBounds>;

    fn contraregions(&self, a: ty::Region, b: ty::Region)
                  -> cres<'tcx, ty::Region>;

    fn regions(&self, a: ty::Region, b: ty::Region) -> cres<'tcx, ty::Region>;

    fn trait_stores(&self,
                    vk: ty::terr_vstore_kind,
                    a: ty::TraitStore,
                    b: ty::TraitStore)
                    -> cres<'tcx, ty::TraitStore> {
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
                  a: &ty::TraitRef<'tcx>,
                  b: &ty::TraitRef<'tcx>)
                  -> cres<'tcx, ty::TraitRef<'tcx>>
    {
        // Different traits cannot be related
        if a.def_id != b.def_id {
            Err(ty::terr_traits(expected_found(self, a.def_id, b.def_id)))
        } else {
            let substs = try!(self.substs(a.def_id, &a.substs, &b.substs));
            Ok(ty::TraitRef { def_id: a.def_id, substs: substs })
        }
    }

    fn binders<T>(&self, a: &ty::Binder<T>, b: &ty::Binder<T>) -> cres<'tcx, ty::Binder<T>>
        where T : Combineable<'tcx>;
    // this must be overridden to do correctly, so as to account for higher-ranked
    // behavior
}

pub trait Combineable<'tcx> : Repr<'tcx> + TypeFoldable<'tcx> {
    fn combine<C:Combine<'tcx>>(combiner: &C, a: &Self, b: &Self) -> cres<'tcx, Self>;
}

impl<'tcx> Combineable<'tcx> for ty::TraitRef<'tcx> {
    fn combine<C:Combine<'tcx>>(combiner: &C,
                                a: &ty::TraitRef<'tcx>,
                                b: &ty::TraitRef<'tcx>)
                                -> cres<'tcx, ty::TraitRef<'tcx>>
    {
        combiner.trait_refs(a, b)
    }
}

impl<'tcx> Combineable<'tcx> for ty::FnSig<'tcx> {
    fn combine<C:Combine<'tcx>>(combiner: &C,
                                a: &ty::FnSig<'tcx>,
                                b: &ty::FnSig<'tcx>)
                                -> cres<'tcx, ty::FnSig<'tcx>>
    {
        combiner.fn_sigs(a, b)
    }
}

#[deriving(Clone)]
pub struct CombineFields<'a, 'tcx: 'a> {
    pub infcx: &'a InferCtxt<'a, 'tcx>,
    pub a_is_expected: bool,
    pub trace: TypeTrace<'tcx>,
}

pub fn expected_found<'tcx, C: Combine<'tcx>, T>(
        this: &C, a: T, b: T) -> ty::expected_found<T> {
    if this.a_is_expected() {
        ty::expected_found {expected: a, found: b}
    } else {
        ty::expected_found {expected: b, found: a}
    }
}

pub fn super_tys<'tcx, C: Combine<'tcx>>(this: &C,
                                         a: Ty<'tcx>,
                                         b: Ty<'tcx>)
                                         -> cres<'tcx, Ty<'tcx>> {

    let tcx = this.infcx().tcx;
    let a_sty = &a.sty;
    let b_sty = &b.sty;
    debug!("super_tys: a_sty={} b_sty={}", a_sty, b_sty);
    return match (a_sty, b_sty) {
      // The "subtype" ought to be handling cases involving var:
      (&ty::ty_infer(TyVar(_)), _) |
      (_, &ty::ty_infer(TyVar(_))) => {
        tcx.sess.bug(
            format!("{}: bot and var types should have been handled ({},{})",
                    this.tag(),
                    a.repr(this.infcx().tcx),
                    b.repr(this.infcx().tcx)).as_slice());
      }

      (&ty::ty_err, _) | (_, &ty::ty_err) => {
          Ok(ty::mk_err())
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
      (&ty::ty_bool, _) |
      (&ty::ty_int(_), _) |
      (&ty::ty_uint(_), _) |
      (&ty::ty_float(_), _) => {
        if a == b {
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
       &ty::ty_trait(ref b_)) => {
          debug!("Trying to match traits {} and {}", a, b);
          let principal = try!(this.binders(&a_.principal, &b_.principal));
          let bounds = try!(this.existential_bounds(a_.bounds, b_.bounds));
          Ok(ty::mk_trait(tcx, principal, bounds))
      }

      (&ty::ty_struct(a_id, ref a_substs), &ty::ty_struct(b_id, ref b_substs))
      if a_id == b_id => {
            let substs = try!(this.substs(a_id, a_substs, b_substs));
            Ok(ty::mk_struct(tcx, a_id, substs))
      }

      (&ty::ty_unboxed_closure(a_id, a_region, ref a_substs),
       &ty::ty_unboxed_closure(b_id, b_region, ref b_substs))
      if a_id == b_id => {
          // All ty_unboxed_closure types with the same id represent
          // the (anonymous) type of the same closure expression. So
          // all of their regions should be equated.
          let region = try!(this.equate().regions(a_region, b_region));
          let substs = try!(this.substs_variances(None, a_substs, b_substs));
          Ok(ty::mk_unboxed_closure(tcx, a_id, region, substs))
      }

      (&ty::ty_uniq(a_inner), &ty::ty_uniq(b_inner)) => {
          let typ = try!(this.tys(a_inner, b_inner));
          Ok(ty::mk_uniq(tcx, typ))
      }

      (&ty::ty_ptr(ref a_mt), &ty::ty_ptr(ref b_mt)) => {
          let mt = try!(this.mts(a_mt, b_mt));
          Ok(ty::mk_ptr(tcx, mt))
      }

      (&ty::ty_rptr(a_r, ref a_mt), &ty::ty_rptr(b_r, ref b_mt)) => {
            let r = try!(this.contraregions(a_r, b_r));
            // FIXME(14985)  If we have mutable references to trait objects, we
            // used to use covariant subtyping. I have preserved this behaviour,
            // even though it is probably incorrect. So don't go down the usual
            // path which would require invariance.
            let mt = match (&a_mt.ty.sty, &b_mt.ty.sty) {
                (&ty::ty_trait(..), &ty::ty_trait(..)) if a_mt.mutbl == b_mt.mutbl => {
                    let ty = try!(this.tys(a_mt.ty, b_mt.ty));
                    ty::mt { ty: ty, mutbl: a_mt.mutbl }
                }
                _ => try!(this.mts(a_mt, b_mt))
            };
            Ok(ty::mk_rptr(tcx, r, mt))
      }

      (&ty::ty_vec(a_t, Some(sz_a)), &ty::ty_vec(b_t, Some(sz_b))) => {
        this.tys(a_t, b_t).and_then(|t| {
            if sz_a == sz_b {
                Ok(ty::mk_vec(tcx, t, Some(sz_a)))
            } else {
                Err(ty::terr_fixed_array_size(expected_found(this, sz_a, sz_b)))
            }
        })
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
            as_.iter().zip(bs.iter())
               .map(|(a, b)| this.tys(*a, *b))
               .collect::<Result<_, _>>()
               .map(|ts| ty::mk_tup(tcx, ts))
        } else if as_.len() != 0 && bs.len() != 0 {
            Err(ty::terr_tuple_size(
                expected_found(this, as_.len(), bs.len())))
        } else {
            Err(ty::terr_sorts(expected_found(this, a, b)))
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
        val: ty::IntVarValue) -> cres<'tcx, Ty<'tcx>>
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
        val: ast::FloatTy) -> cres<'tcx, Ty<'tcx>>
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
                       a_ty: Ty<'tcx>,
                       dir: RelationDir,
                       b_vid: ty::TyVid)
                       -> cres<'tcx, ()>
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

    /// Attempts to generalize `ty` for the type variable `for_vid`.  This checks for cycle -- that
    /// is, whether the type `ty` references `for_vid`. If `make_region_vars` is true, it will also
    /// replace all regions with fresh variables. Returns `ty_err` in the case of a cycle, `Ok`
    /// otherwise.
    fn generalize(&self,
                  ty: Ty<'tcx>,
                  for_vid: ty::TyVid,
                  make_region_vars: bool)
                  -> cres<'tcx, Ty<'tcx>>
    {
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

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        // Check to see whether the type we are genealizing references
        // `vid`. At the same time, also update any type variables to
        // the values that they are bound to. This is needed to truly
        // check for cycles, but also just makes things readable.
        //
        // (In particular, you could have something like `$0 = Box<$1>`
        //  where `$1` has already been instantiated with `Box<$0>`)
        match t.sty {
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
            // Never make variables for regions bound within the type itself.
            ty::ReLateBound(..) => { return r; }

            // Early-bound regions should really have been substituted away before
            // we get to this point.
            ty::ReEarlyBound(..) => {
                self.tcx().sess.span_bug(
                    self.span,
                    format!("Encountered early bound region when generalizing: {}",
                            r.repr(self.tcx()))[]);
            }

            // Always make a fresh region variable for skolemized regions;
            // the higher-ranked decision procedures rely on this.
            ty::ReInfer(ty::ReSkolemized(..)) => { }

            // For anything else, we make a region variable, unless we
            // are *equating*, in which case it's just wasteful.
            ty::ReEmpty |
            ty::ReStatic |
            ty::ReScope(..) |
            ty::ReInfer(ty::ReVar(..)) |
            ty::ReFree(..) => {
                if !self.make_region_vars {
                    return r;
                }
            }
        }

        // FIXME: This is non-ideal because we don't give a
        // very descriptive origin for this region variable.
        self.infcx.next_region_var(MiscVariable(self.span))
    }
}


