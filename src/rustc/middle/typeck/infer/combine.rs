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

use to_str::ToStr;
use ty::{FnTyBase, FnMeta, FnSig};
use syntax::ast::Onceness;

trait combine {
    fn infcx() -> infer_ctxt;
    fn tag() -> ~str;
    fn a_is_expected() -> bool;

    fn sub() -> Sub;
    fn lub() -> Lub;
    fn glb() -> Glb;

    fn mts(a: ty::mt, b: ty::mt) -> cres<ty::mt>;
    fn contratys(a: ty::t, b: ty::t) -> cres<ty::t>;
    fn tys(a: ty::t, b: ty::t) -> cres<ty::t>;
    fn tps(as_: &[ty::t], bs: &[ty::t]) -> cres<~[ty::t]>;
    fn self_tys(a: Option<ty::t>, b: Option<ty::t>) -> cres<Option<ty::t>>;
    fn substs(did: ast::def_id, as_: &ty::substs,
              bs: &ty::substs) -> cres<ty::substs>;
    fn fns(a: &ty::FnTy, b: &ty::FnTy) -> cres<ty::FnTy>;
    fn fn_sigs(a: &ty::FnSig, b: &ty::FnSig) -> cres<ty::FnSig>;
    fn fn_metas(a: &ty::FnMeta, b: &ty::FnMeta) -> cres<ty::FnMeta>;
    fn flds(a: ty::field, b: ty::field) -> cres<ty::field>;
    fn modes(a: ast::mode, b: ast::mode) -> cres<ast::mode>;
    fn args(a: ty::arg, b: ty::arg) -> cres<ty::arg>;
    fn protos(p1: ty::fn_proto, p2: ty::fn_proto) -> cres<ty::fn_proto>;
    fn ret_styles(r1: ret_style, r2: ret_style) -> cres<ret_style>;
    fn purities(a: purity, b: purity) -> cres<purity>;
    fn oncenesses(a: Onceness, b: Onceness) -> cres<Onceness>;
    fn contraregions(a: ty::Region, b: ty::Region) -> cres<ty::Region>;
    fn regions(a: ty::Region, b: ty::Region) -> cres<ty::Region>;
    fn vstores(vk: ty::terr_vstore_kind,
               a: ty::vstore, b: ty::vstore) -> cres<ty::vstore>;
}

struct combine_fields {
    infcx: infer_ctxt,
    a_is_expected: bool,
    span: span,
}

fn expected_found<C: combine,T>(
    self: &C, +a: T, +b: T) -> ty::expected_found<T> {

    if self.a_is_expected() {
        ty::expected_found {expected: move a, found: move b}
    } else {
        ty::expected_found {expected: move b, found: move a}
    }
}

fn eq_tys<C: combine>(self: &C, a: ty::t, b: ty::t) -> ures {
    let suber = self.sub();
    do self.infcx().try {
        do suber.tys(a, b).chain |_ok| {
            suber.contratys(a, b)
        }.to_ures()
    }
}

fn eq_regions<C: combine>(self: &C, a: ty::Region, b: ty::Region) -> ures {
    debug!("eq_regions(%s, %s)",
           a.to_str(self.infcx()),
           b.to_str(self.infcx()));
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

fn eq_opt_regions<C:combine>(
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
                 a.to_str(self.infcx()),
                 b.to_str(self.infcx())));
      }
    }
}

fn super_substs<C:combine>(
    self: &C, did: ast::def_id,
    a: &ty::substs, b: &ty::substs) -> cres<ty::substs> {

    fn relate_region_param<C:combine>(
        self: &C,
        did: ast::def_id,
        a: Option<ty::Region>,
        b: Option<ty::Region>)
        -> cres<Option<ty::Region>>
    {
        let polyty = ty::lookup_item_type(self.infcx().tcx, did);
        match (polyty.region_param, a, b) {
          (None, None, None) => {
            Ok(None)
          }
          (Some(ty::rv_invariant), Some(a), Some(b)) => {
            do eq_regions(self, a, b).then {
                Ok(Some(a))
            }
          }
          (Some(ty::rv_covariant), Some(a), Some(b)) => {
            do self.regions(a, b).chain |r| {
                Ok(Some(r))
            }
          }
          (Some(ty::rv_contravariant), Some(a), Some(b)) => {
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
                      a.to_str(self.infcx()),
                      b.to_str(self.infcx()),
                      polyty.region_param));
          }
        }
    }

    do self.tps(a.tps, b.tps).chain |tps| {
        do self.self_tys(a.self_ty, b.self_ty).chain |self_ty| {
            do relate_region_param(self, did,
                                   a.self_r, b.self_r).chain |self_r|
            {
                Ok({self_r: self_r, self_ty: self_ty, tps: tps})
            }
        }
    }
}

fn super_tps<C:combine>(
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

fn super_self_tys<C:combine>(
    self: &C, a: Option<ty::t>, b: Option<ty::t>) -> cres<Option<ty::t>> {

    // Note: the self type parameter is (currently) always treated as
    // *invariant* (otherwise the type system would be unsound).

    match (a, b) {
      (None, None) => {
        Ok(None)
      }
      (Some(a), Some(b)) => {
        eq_tys(self, a, b).then(|| Ok(Some(a)) )
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

fn super_flds<C:combine>(
    self: &C, a: ty::field, b: ty::field) -> cres<ty::field> {

    if a.ident == b.ident {
        self.mts(a.mt, b.mt)
            .chain(|mt| Ok({ident: a.ident, mt: mt}) )
            .chain_err(|e| Err(ty::terr_in_field(@e, a.ident)) )
    } else {
        Err(ty::terr_record_fields(
            expected_found(self, a.ident, b.ident)))
    }
}

fn super_modes<C:combine>(
    self: &C, a: ast::mode, b: ast::mode)
    -> cres<ast::mode> {

    let tcx = self.infcx().tcx;
    ty::unify_mode(tcx, expected_found(self, a, b))
}

fn super_args<C:combine>(
    self: &C, a: ty::arg, b: ty::arg)
    -> cres<ty::arg> {

    do self.modes(a.mode, b.mode).chain |m| {
        do self.contratys(a.ty, b.ty).chain |t| {
            Ok({mode: m, ty: t})
        }
    }
}

fn super_vstores<C:combine>(
    self: &C, vk: ty::terr_vstore_kind,
    a: ty::vstore, b: ty::vstore) -> cres<ty::vstore> {

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

fn super_fn_metas<C:combine>(
    self: &C, a_f: &ty::FnMeta, b_f: &ty::FnMeta) -> cres<ty::FnMeta>
{
    do self.protos(a_f.proto, b_f.proto).chain |p| {
        do self.ret_styles(a_f.ret_style, b_f.ret_style).chain |rs| {
            do self.purities(a_f.purity, b_f.purity).chain |purity| {
                do self.oncenesses(a_f.onceness, b_f.onceness).chain
                        |onceness| {
                    Ok(FnMeta {purity: purity,
                               proto: p,
                               onceness: onceness,
                               bounds: a_f.bounds, // XXX: This is wrong!
                               ret_style: rs})
                }
            }
        }
    }
}

fn super_fn_sigs<C:combine>(
    self: &C, a_f: &ty::FnSig, b_f: &ty::FnSig) -> cres<ty::FnSig>
{
    fn argvecs<C:combine>(self: &C, a_args: ~[ty::arg],
                          b_args: ~[ty::arg]) -> cres<~[ty::arg]> {

        if vec::same_length(a_args, b_args) {
            map_vec2(a_args, b_args, |a, b| self.args(*a, *b))
        } else {
            Err(ty::terr_arg_count)
        }
    }

    do argvecs(self, a_f.inputs, b_f.inputs).chain |inputs| {
        do self.tys(a_f.output, b_f.output).chain |output| {
            Ok(FnSig {inputs: inputs, output: output})
        }
    }
}

fn super_fns<C:combine>(
    self: &C, a_f: &ty::FnTy, b_f: &ty::FnTy) -> cres<ty::FnTy>
{
    do self.fn_metas(&a_f.meta, &b_f.meta).chain |m| {
        do self.fn_sigs(&a_f.sig, &b_f.sig).chain |s| {
            Ok(FnTyBase {meta: m, sig: s})
        }
    }
}

fn super_tys<C:combine>(
    self: &C, a: ty::t, b: ty::t) -> cres<ty::t> {

    let tcx = self.infcx().tcx;
    match (ty::get(a).sty, ty::get(b).sty) {
      // The "subtype" ought to be handling cases involving bot or var:
      (ty::ty_bot, _) |
      (_, ty::ty_bot) |
      (ty::ty_infer(TyVar(_)), _) |
      (_, ty::ty_infer(TyVar(_))) => {
        tcx.sess.bug(
            fmt!("%s: bot and var types should have been handled (%s,%s)",
                 self.tag(),
                 a.to_str(self.infcx()),
                 b.to_str(self.infcx())));
      }

      // Relate integral variables to other types
      (ty::ty_infer(IntVar(a_id)), ty::ty_infer(IntVar(b_id))) => {
        self.infcx().int_vars(a_id, b_id).then(|| Ok(a) )
      }
      (ty::ty_infer(IntVar(a_id)), ty::ty_int(_)) |
      (ty::ty_infer(IntVar(a_id)), ty::ty_uint(_)) => {
        self.infcx().int_var_sub_t(a_id, b).then(|| Ok(a) )
      }
      (ty::ty_int(_), ty::ty_infer(IntVar(b_id))) |
      (ty::ty_uint(_), ty::ty_infer(IntVar(b_id))) => {
        self.infcx().t_sub_int_var(a, b_id).then(|| Ok(a) )
      }

      (ty::ty_int(_), _) |
      (ty::ty_uint(_), _) |
      (ty::ty_float(_), _) => {
        let as_ = ty::get(a).sty;
        let bs = ty::get(b).sty;
        if as_ == bs {
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

      (ty::ty_param(a_p), ty::ty_param(b_p)) if a_p.idx == b_p.idx => {
        Ok(a)
      }

      (ty::ty_enum(a_id, ref a_substs),
       ty::ty_enum(b_id, ref b_substs))
      if a_id == b_id => {
        do self.substs(a_id, a_substs, b_substs).chain |substs| {
            Ok(ty::mk_enum(tcx, a_id, substs))
        }
      }

      (ty::ty_trait(a_id, ref a_substs, a_vstore),
       ty::ty_trait(b_id, ref b_substs, b_vstore))
      if a_id == b_id => {
        do self.substs(a_id, a_substs, b_substs).chain |substs| {
            do self.vstores(ty::terr_trait, a_vstore, b_vstore).chain |vs| {
                Ok(ty::mk_trait(tcx, a_id, substs, vs))
            }
        }
      }

      (ty::ty_class(a_id, ref a_substs), ty::ty_class(b_id, ref b_substs))
      if a_id == b_id => {
        do self.substs(a_id, a_substs, b_substs).chain |substs| {
            Ok(ty::mk_class(tcx, a_id, substs))
        }
      }

      (ty::ty_box(a_mt), ty::ty_box(b_mt)) => {
        do self.mts(a_mt, b_mt).chain |mt| {
            Ok(ty::mk_box(tcx, mt))
        }
      }

      (ty::ty_uniq(a_mt), ty::ty_uniq(b_mt)) => {
        do self.mts(a_mt, b_mt).chain |mt| {
            Ok(ty::mk_uniq(tcx, mt))
        }
      }

      (ty::ty_ptr(a_mt), ty::ty_ptr(b_mt)) => {
        do self.mts(a_mt, b_mt).chain |mt| {
            Ok(ty::mk_ptr(tcx, mt))
        }
      }

      (ty::ty_rptr(a_r, a_mt), ty::ty_rptr(b_r, b_mt)) => {
        do self.contraregions(a_r, b_r).chain |r| {
            do self.mts(a_mt, b_mt).chain |mt| {
                Ok(ty::mk_rptr(tcx, r, mt))
            }
        }
      }

      (ty::ty_evec(a_mt, vs_a), ty::ty_evec(b_mt, vs_b)) => {
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

      (ty::ty_rec(as_), ty::ty_rec(bs)) => {
        if vec::same_length(as_, bs) {
            map_vec2(as_, bs, |a,b| {
                self.flds(*a, *b)
            }).chain(|flds| Ok(ty::mk_rec(tcx, flds)) )
        } else {
            Err(ty::terr_record_size(expected_found(self, as_.len(),
                                                    bs.len())))
        }
      }

      (ty::ty_tup(as_), ty::ty_tup(bs)) => {
        if vec::same_length(as_, bs) {
            map_vec2(as_, bs, |a, b| self.tys(*a, *b) )
                .chain(|ts| Ok(ty::mk_tup(tcx, ts)) )
        } else {
            Err(ty::terr_tuple_size(
                expected_found(self, as_.len(), bs.len())))
        }
      }

      (ty::ty_fn(ref a_fty), ty::ty_fn(ref b_fty)) => {
        do self.fns(a_fty, b_fty).chain |fty| {
            Ok(ty::mk_fn(tcx, fty))
        }
      }

      _ => Err(ty::terr_sorts(expected_found(self, a, b)))
    }
}

