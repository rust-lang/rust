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

import to_str::to_str;

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
    fn tps(as: &[ty::t], bs: &[ty::t]) -> cres<~[ty::t]>;
    fn self_tys(a: Option<ty::t>, b: Option<ty::t>) -> cres<Option<ty::t>>;
    fn substs(did: ast::def_id, as: &ty::substs,
              bs: &ty::substs) -> cres<ty::substs>;
    fn fns(a: &ty::fn_ty, b: &ty::fn_ty) -> cres<ty::fn_ty>;
    fn flds(a: ty::field, b: ty::field) -> cres<ty::field>;
    fn modes(a: ast::mode, b: ast::mode) -> cres<ast::mode>;
    fn args(a: ty::arg, b: ty::arg) -> cres<ty::arg>;
    fn protos(p1: ty::fn_proto, p2: ty::fn_proto) -> cres<ty::fn_proto>;
    fn ret_styles(r1: ret_style, r2: ret_style) -> cres<ret_style>;
    fn purities(a: purity, b: purity) -> cres<purity>;
    fn contraregions(a: ty::region, b: ty::region) -> cres<ty::region>;
    fn regions(a: ty::region, b: ty::region) -> cres<ty::region>;
    fn vstores(vk: ty::terr_vstore_kind,
               a: ty::vstore, b: ty::vstore) -> cres<ty::vstore>;
}

struct combine_fields {
    infcx: infer_ctxt;
    a_is_expected: bool;
    span: span;
}

fn expected_found<C: combine,T>(
    self: &C, +a: T, +b: T) -> ty::expected_found<T> {

    if self.a_is_expected() {
        ty::expected_found {expected: a, found: b}
    } else {
        ty::expected_found {expected: b, found: a}
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

fn eq_regions<C: combine>(self: &C, a: ty::region, b: ty::region) -> ures {
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
                err(ty::terr_regions_not_same(a1, b1)),
              _ => err(e)
            }
        }).to_ures()
    }
}

fn eq_opt_regions<C:combine>(
    self: &C,
    a: Option<ty::region>,
    b: Option<ty::region>) -> cres<Option<ty::region>> {

    match (a, b) {
      (None, None) => {
        ok(None)
      }
      (Some(a), Some(b)) => {
        do eq_regions(self, a, b).then {
            ok(Some(a))
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
        a: Option<ty::region>,
        b: Option<ty::region>)
        -> cres<Option<ty::region>>
    {
        let polyty = ty::lookup_item_type(self.infcx().tcx, did);
        match (polyty.region_param, a, b) {
          (None, None, None) => {
            ok(None)
          }
          (Some(ty::rv_invariant), Some(a), Some(b)) => {
            do eq_regions(self, a, b).then {
                ok(Some(a))
            }
          }
          (Some(ty::rv_covariant), Some(a), Some(b)) => {
            do self.regions(a, b).chain |r| {
                ok(Some(r))
            }
          }
          (Some(ty::rv_contravariant), Some(a), Some(b)) => {
            do self.contraregions(a, b).chain |r| {
                ok(Some(r))
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
                ok({self_r: self_r, self_ty: self_ty, tps: tps})
            }
        }
    }
}

fn super_tps<C:combine>(
    self: &C, as: &[ty::t], bs: &[ty::t]) -> cres<~[ty::t]> {

    // Note: type parameters are always treated as *invariant*
    // (otherwise the type system would be unsound).  In the
    // future we could allow type parameters to declare a
    // variance.

    if vec::same_length(as, bs) {
        iter_vec2(as, bs, |a, b| {
            eq_tys(self, a, b)
        }).then(|| ok(as.to_vec()) )
    } else {
        err(ty::terr_ty_param_size(
            expected_found(self, as.len(), bs.len())))
    }
}

fn super_self_tys<C:combine>(
    self: &C, a: Option<ty::t>, b: Option<ty::t>) -> cres<Option<ty::t>> {

    // Note: the self type parameter is (currently) always treated as
    // *invariant* (otherwise the type system would be unsound).

    match (a, b) {
      (None, None) => {
        ok(None)
      }
      (Some(a), Some(b)) => {
        eq_tys(self, a, b).then(|| ok(Some(a)) )
      }
      (None, Some(_)) |
      (Some(_), None) => {
        // I think it should never happen that we unify two substs and
        // one of them has a self_ty and one doesn't...? I could be
        // wrong about this.
        err(ty::terr_self_substs)
      }
    }
}

fn super_flds<C:combine>(
    self: &C, a: ty::field, b: ty::field) -> cres<ty::field> {

    if a.ident == b.ident {
        self.mts(a.mt, b.mt)
            .chain(|mt| ok({ident: a.ident, mt: mt}) )
            .chain_err(|e| err(ty::terr_in_field(@e, a.ident)) )
    } else {
        err(ty::terr_record_fields(
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
            ok({mode: m, ty: t})
        }
    }
}

fn super_vstores<C:combine>(
    self: &C, vk: ty::terr_vstore_kind,
    a: ty::vstore, b: ty::vstore) -> cres<ty::vstore> {

    match (a, b) {
      (ty::vstore_slice(a_r), ty::vstore_slice(b_r)) => {
        do self.contraregions(a_r, b_r).chain |r| {
            ok(ty::vstore_slice(r))
        }
      }

      _ if a == b => {
        ok(a)
      }

      _ => {
        err(ty::terr_vstores_differ(vk, expected_found(self, a, b)))
      }
    }
}

fn super_fns<C:combine>(
    self: &C, a_f: &ty::fn_ty, b_f: &ty::fn_ty) -> cres<ty::fn_ty> {

    fn argvecs<C:combine>(self: &C, a_args: ~[ty::arg],
                          b_args: ~[ty::arg]) -> cres<~[ty::arg]> {

        if vec::same_length(a_args, b_args) {
            map_vec2(a_args, b_args, |a, b| self.args(a, b) )
        } else {
            err(ty::terr_arg_count)
        }
    }

    do self.protos(a_f.proto, b_f.proto).chain |p| {
        do self.ret_styles(a_f.ret_style, b_f.ret_style).chain |rs| {
            do argvecs(self, a_f.inputs, b_f.inputs).chain |inputs| {
                do self.tys(a_f.output, b_f.output).chain |output| {
                    do self.purities(a_f.purity, b_f.purity).chain |purity| {
                    // FIXME: uncomment if #2588 doesn't get accepted:
                    // self.infcx().constrvecs(a_f.constraints,
                    //                         b_f.constraints).then {||
                        ok({purity: purity,
                            proto: p,
                            bounds: a_f.bounds, // XXX: This is wrong!
                            inputs: inputs,
                            output: output,
                            ret_style: rs})
                    // }
                    }
                }
            }
        }
    }
}

fn super_tys<C:combine>(
    self: &C, a: ty::t, b: ty::t) -> cres<ty::t> {

    let tcx = self.infcx().tcx;
    match (ty::get(a).struct, ty::get(b).struct) {
      // The "subtype" ought to be handling cases involving bot or var:
      (ty::ty_bot, _) |
      (_, ty::ty_bot) |
      (ty::ty_var(_), _) |
      (_, ty::ty_var(_)) => {
        tcx.sess.bug(
            fmt!("%s: bot and var types should have been handled (%s,%s)",
                 self.tag(),
                 a.to_str(self.infcx()),
                 b.to_str(self.infcx())));
      }

      // Relate integral variables to other types
      (ty::ty_var_integral(a_id), ty::ty_var_integral(b_id)) => {
        self.infcx().vars_integral(a_id, b_id).then(|| ok(a) )
      }
      (ty::ty_var_integral(a_id), ty::ty_int(_)) |
      (ty::ty_var_integral(a_id), ty::ty_uint(_)) => {
        self.infcx().var_integral_sub_t(a_id, b).then(|| ok(a) )
      }
      (ty::ty_int(_), ty::ty_var_integral(b_id)) |
      (ty::ty_uint(_), ty::ty_var_integral(b_id)) => {
        self.infcx().t_sub_var_integral(a, b_id).then(|| ok(a) )
      }

      (ty::ty_int(_), _) |
      (ty::ty_uint(_), _) |
      (ty::ty_float(_), _) => {
        let as = ty::get(a).struct;
        let bs = ty::get(b).struct;
        if as == bs {
            ok(a)
        } else {
            err(ty::terr_sorts(expected_found(self, a, b)))
        }
      }

      (ty::ty_nil, _) |
      (ty::ty_bool, _) => {
        let cfg = tcx.sess.targ_cfg;
        if ty::mach_sty(cfg, a) == ty::mach_sty(cfg, b) {
            ok(a)
        } else {
            err(ty::terr_sorts(expected_found(self, a, b)))
        }
      }

      (ty::ty_param(a_p), ty::ty_param(b_p)) if a_p.idx == b_p.idx => {
        ok(a)
      }

      (ty::ty_enum(a_id, ref a_substs),
       ty::ty_enum(b_id, ref b_substs))
      if a_id == b_id => {
        do self.substs(a_id, a_substs, b_substs).chain |substs| {
            ok(ty::mk_enum(tcx, a_id, substs))
        }
      }

      (ty::ty_trait(a_id, ref a_substs, a_vstore),
       ty::ty_trait(b_id, ref b_substs, b_vstore))
      if a_id == b_id => {
        do self.substs(a_id, a_substs, b_substs).chain |substs| {
            do self.vstores(ty::terr_trait, a_vstore, b_vstore).chain |vs| {
                ok(ty::mk_trait(tcx, a_id, substs, vs))
            }
        }
      }

      (ty::ty_class(a_id, ref a_substs), ty::ty_class(b_id, ref b_substs))
      if a_id == b_id => {
        do self.substs(a_id, a_substs, b_substs).chain |substs| {
            ok(ty::mk_class(tcx, a_id, substs))
        }
      }

      (ty::ty_box(a_mt), ty::ty_box(b_mt)) => {
        do self.mts(a_mt, b_mt).chain |mt| {
            ok(ty::mk_box(tcx, mt))
        }
      }

      (ty::ty_uniq(a_mt), ty::ty_uniq(b_mt)) => {
        do self.mts(a_mt, b_mt).chain |mt| {
            ok(ty::mk_uniq(tcx, mt))
        }
      }

      (ty::ty_ptr(a_mt), ty::ty_ptr(b_mt)) => {
        do self.mts(a_mt, b_mt).chain |mt| {
            ok(ty::mk_ptr(tcx, mt))
        }
      }

      (ty::ty_rptr(a_r, a_mt), ty::ty_rptr(b_r, b_mt)) => {
        do self.contraregions(a_r, b_r).chain |r| {
            do self.mts(a_mt, b_mt).chain |mt| {
                ok(ty::mk_rptr(tcx, r, mt))
            }
        }
      }

      (ty::ty_evec(a_mt, vs_a), ty::ty_evec(b_mt, vs_b)) => {
        do self.mts(a_mt, b_mt).chain |mt| {
            do self.vstores(ty::terr_vec, vs_a, vs_b).chain |vs| {
                ok(ty::mk_evec(tcx, mt, vs))
            }
        }
      }

      (ty::ty_estr(vs_a), ty::ty_estr(vs_b)) => {
        do self.vstores(ty::terr_str, vs_a, vs_b).chain |vs| {
            ok(ty::mk_estr(tcx,vs))
        }
      }

      (ty::ty_rec(as), ty::ty_rec(bs)) => {
        if vec::same_length(as, bs) {
            map_vec2(as, bs, |a,b| {
                self.flds(a, b)
            }).chain(|flds| ok(ty::mk_rec(tcx, flds)) )
        } else {
            err(ty::terr_record_size(expected_found(self, as.len(),
                                                    bs.len())))
        }
      }

      (ty::ty_tup(as), ty::ty_tup(bs)) => {
        if vec::same_length(as, bs) {
            map_vec2(as, bs, |a, b| self.tys(a, b) )
                .chain(|ts| ok(ty::mk_tup(tcx, ts)) )
        } else {
            err(ty::terr_tuple_size(expected_found(self, as.len(), bs.len())))
        }
      }

      (ty::ty_fn(ref a_fty), ty::ty_fn(ref b_fty)) => {
        do self.fns(a_fty, b_fty).chain |fty| {
            ok(ty::mk_fn(tcx, fty))
        }
      }

      _ => err(ty::terr_sorts(expected_found(self, a, b)))
    }
}

