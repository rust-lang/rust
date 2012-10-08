/*!
 * Conversion from AST representation of types to the ty.rs
 * representation.  The main routine here is `ast_ty_to_ty()`: each use
 * is parameterized by an instance of `ast_conv` and a `region_scope`.
 *
 * The parameterization of `ast_ty_to_ty()` is because it behaves
 * somewhat differently during the collect and check phases, particularly
 * with respect to looking up the types of top-level items.  In the
 * collect phase, the crate context is used as the `ast_conv` instance;
 * in this phase, the `get_item_ty()` function triggers a recursive call
 * to `ty_of_item()` (note that `ast_ty_to_ty()` will detect recursive
 * types and report an error).  In the check phase, when the @fn_ctxt is
 * used as the `ast_conv`, `get_item_ty()` just looks up the item type in
 * `tcx.tcache`.
 *
 * The `region_scope` trait controls how region references are
 * handled.  It has two methods which are used to resolve anonymous
 * region references (e.g., `&T`) and named region references (e.g.,
 * `&a.T`).  There are numerous region scopes that can be used, but most
 * commonly you want either `empty_rscope`, which permits only the static
 * region, or `type_rscope`, which permits the self region if the type in
 * question is parameterized by a region.
 *
 * Unlike the `ast_conv` trait, the region scope can change as we descend
 * the type.  This is to accommodate the fact that (a) fn types are binding
 * scopes and (b) the default region may change.  To understand case (a),
 * consider something like:
 *
 *   type foo = { x: &a.int, y: fn(&a.int) }
 *
 * The type of `x` is an error because there is no region `a` in scope.
 * In the type of `y`, however, region `a` is considered a bound region
 * as it does not already appear in scope.
 *
 * Case (b) says that if you have a type:
 *   type foo/& = ...;
 *   type bar = fn(&foo, &a.foo)
 * The fully expanded version of type bar is:
 *   type bar = fn(&foo/&, &a.foo/&a)
 * Note that the self region for the `foo` defaulted to `&` in the first
 * case but `&a` in the second.  Basically, defaults that appear inside
 * an rptr (`&r.T`) use the region `r` that appears in the rptr.
 */

use check::fn_ctxt;
use rscope::{anon_rscope, binding_rscope, empty_rscope, in_anon_rscope};
use rscope::{in_binding_rscope, region_scope, type_rscope};
use ty::{FnTyBase, FnMeta, FnSig};

trait ast_conv {
    fn tcx() -> ty::ctxt;
    fn ccx() -> @crate_ctxt;
    fn get_item_ty(id: ast::def_id) -> ty::ty_param_bounds_and_ty;

    // what type should we use when a type is omitted?
    fn ty_infer(span: span) -> ty::t;
}

fn get_region_reporting_err(tcx: ty::ctxt,
                            span: span,
                            res: Result<ty::region, ~str>) -> ty::region {

    match res {
      result::Ok(r) => r,
      result::Err(e) => {
        tcx.sess.span_err(span, e);
        ty::re_static
      }
    }
}

fn ast_region_to_region<AC: ast_conv, RS: region_scope Copy Owned>(
    self: AC, rscope: RS, span: span, a_r: @ast::region) -> ty::region {

    let res = match a_r.node {
        ast::re_static => Ok(ty::re_static),
        ast::re_anon => rscope.anon_region(span),
        ast::re_self => rscope.self_region(span),
        ast::re_named(id) => rscope.named_region(span, id)
    };

    get_region_reporting_err(self.tcx(), span, res)
}

fn ast_path_to_substs_and_ty<AC: ast_conv, RS: region_scope Copy Owned>(
    self: AC, rscope: RS, did: ast::def_id,
    path: @ast::path) -> ty_param_substs_and_ty {

    let tcx = self.tcx();
    let {bounds: decl_bounds, region_param: decl_rp, ty: decl_ty} =
        self.get_item_ty(did);

    debug!("ast_path_to_substs_and_ty: did=%? decl_rp=%?",
           did, decl_rp);

    // If the type is parameterized by the self region, then replace self
    // region with the current anon region binding (in other words,
    // whatever & would get replaced with).
    let self_r = match (decl_rp, path.rp) {
      (None, None) => {
        None
      }
      (None, Some(_)) => {
        tcx.sess.span_err(
            path.span,
            fmt!("no region bound is allowed on `%s`, \
                  which is not declared as containing region pointers",
                 ty::item_path_str(tcx, did)));
        None
      }
      (Some(_), None) => {
        let res = rscope.anon_region(path.span);
        let r = get_region_reporting_err(self.tcx(), path.span, res);
        Some(r)
      }
      (Some(_), Some(r)) => {
        Some(ast_region_to_region(self, rscope, path.span, r))
      }
    };

    // Convert the type parameters supplied by the user.
    if !vec::same_length(*decl_bounds, path.types) {
        self.tcx().sess.span_fatal(
            path.span,
            fmt!("wrong number of type arguments: expected %u but found %u",
                 (*decl_bounds).len(), path.types.len()));
    }
    let tps = path.types.map(|a_t| ast_ty_to_ty(self, rscope, *a_t));

    let substs = {self_r:self_r, self_ty:None, tps:tps};
    {substs: substs, ty: ty::subst(tcx, &substs, decl_ty)}
}

fn ast_path_to_ty<AC: ast_conv, RS: region_scope Copy Owned>(
    self: AC,
    rscope: RS,
    did: ast::def_id,
    path: @ast::path,
    path_id: ast::node_id) -> ty_param_substs_and_ty {

    // Look up the polytype of the item and then substitute the provided types
    // for any type/region parameters.
    let tcx = self.tcx();
    let {substs: substs, ty: ty} =
        ast_path_to_substs_and_ty(self, rscope, did, path);
    write_ty_to_tcx(tcx, path_id, ty);
    write_substs_to_tcx(tcx, path_id, substs.tps);
    return {substs: substs, ty: ty};
}

const NO_REGIONS: uint = 1u;
const NO_TPS: uint = 2u;

// Parses the programmer's textual representation of a type into our
// internal notion of a type. `getter` is a function that returns the type
// corresponding to a definition ID:
fn ast_ty_to_ty<AC: ast_conv, RS: region_scope Copy Owned>(
    self: AC, rscope: RS, &&ast_ty: @ast::ty) -> ty::t {

    fn ast_mt_to_mt<AC: ast_conv, RS: region_scope Copy Owned>(
        self: AC, rscope: RS, mt: ast::mt) -> ty::mt {

        return {ty: ast_ty_to_ty(self, rscope, mt.ty), mutbl: mt.mutbl};
    }

    // Handle @, ~, and & being able to mean estrs and evecs.
    // If a_seq_ty is a str or a vec, make it an estr/evec.
    // Also handle function sigils and first-class trait types.
    fn mk_maybe_vstore<AC: ast_conv, RS: region_scope Copy Owned>(
        self: AC, rscope: RS, a_seq_ty: ast::mt, vst: ty::vstore,
        span: span, constr: fn(ty::mt) -> ty::t) -> ty::t {

        let tcx = self.tcx();

        match a_seq_ty.ty.node {
          // to convert to an e{vec,str}, there can't be a mutability argument
          _ if a_seq_ty.mutbl != ast::m_imm => (),
          ast::ty_vec(mt) => {
            return ty::mk_evec(tcx, ast_mt_to_mt(self, rscope, mt), vst);
          }
          ast::ty_path(path, id) => {
            match tcx.def_map.find(id) {
              Some(ast::def_prim_ty(ast::ty_str)) => {
                check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                return ty::mk_estr(tcx, vst);
              }
              Some(ast::def_ty(type_def_id)) => {
                let result = ast_path_to_substs_and_ty(self, rscope,
                                                       type_def_id, path);
                match ty::get(result.ty).sty {
                    ty::ty_trait(trait_def_id, substs, _) => {
                        match vst {
                            ty::vstore_box | ty::vstore_slice(*) => {}
                            _ => {
                                tcx.sess.span_unimpl(path.span,
                                                     ~"`~trait` is \
                                                       unimplemented; use \
                                                       `@trait` instead for \
                                                       now");
                            }
                        }
                        return ty::mk_trait(tcx, trait_def_id, substs, vst);
                    }
                    _ => {}
                }
              }
              _ => ()
            }
          }
          ast::ty_fn(ast::proto_block, purity, ast_bounds, ast_fn_decl) => {
            let new_proto;
            match vst {
                ty::vstore_fixed(_) => {
                    tcx.sess.span_err(span, ~"fixed-length functions are not \
                                              allowed");
                    new_proto = ast::proto_block;
                }
                ty::vstore_uniq => new_proto = ast::proto_uniq,
                ty::vstore_box => new_proto = ast::proto_box,
                ty::vstore_slice(_) => new_proto = ast::proto_block
            }

            // Run through the normal function type conversion process.
            let bounds = collect::compute_bounds(self.ccx(), ast_bounds);
            let fn_decl = ty_of_fn_decl(self, rscope, new_proto, purity,
                                        bounds,
                                        ast_fn_decl, None, span);
            return ty::mk_fn(tcx, fn_decl);
          }
          _ => ()
        }

        let seq_ty = ast_mt_to_mt(self, rscope, a_seq_ty);
        return constr(seq_ty);
    }

    fn check_path_args(tcx: ty::ctxt,
                       path: @ast::path,
                       flags: uint) {
        if (flags & NO_TPS) != 0u {
            if path.types.len() > 0u {
                tcx.sess.span_err(
                    path.span,
                    ~"type parameters are not allowed on this type");
            }
        }

        if (flags & NO_REGIONS) != 0u {
            if path.rp.is_some() {
                tcx.sess.span_err(
                    path.span,
                    ~"region parameters are not allowed on this type");
            }
        }
    }

    let tcx = self.tcx();

    match tcx.ast_ty_to_ty_cache.find(ast_ty) {
      Some(ty::atttce_resolved(ty)) => return ty,
      Some(ty::atttce_unresolved) => {
        tcx.sess.span_fatal(ast_ty.span, ~"illegal recursive type; \
                                          insert an enum in the cycle, \
                                          if this is desired");
      }
      None => { /* go on */ }
    }

    tcx.ast_ty_to_ty_cache.insert(ast_ty, ty::atttce_unresolved);
    let typ = match ast_ty.node {
      ast::ty_nil => ty::mk_nil(tcx),
      ast::ty_bot => ty::mk_bot(tcx),
      ast::ty_box(mt) => {
        mk_maybe_vstore(self, rscope, mt, ty::vstore_box, ast_ty.span,
                        |tmt| ty::mk_box(tcx, tmt))
      }
      ast::ty_uniq(mt) => {
        mk_maybe_vstore(self, rscope, mt, ty::vstore_uniq, ast_ty.span,
                        |tmt| ty::mk_uniq(tcx, tmt))
      }
      ast::ty_vec(mt) => {
        tcx.sess.span_err(ast_ty.span,
                          ~"bare `[]` is not a type");
        // return /something/ so they can at least get more errors
        ty::mk_evec(tcx, ast_mt_to_mt(self, rscope, mt),
                    ty::vstore_uniq)
      }
      ast::ty_ptr(mt) => {
        ty::mk_ptr(tcx, ast_mt_to_mt(self, rscope, mt))
      }
      ast::ty_rptr(region, mt) => {
        let r = ast_region_to_region(self, rscope, ast_ty.span, region);
        mk_maybe_vstore(self,
                        in_anon_rscope(rscope, r),
                        mt,
                        ty::vstore_slice(r),
                        ast_ty.span,
                        |tmt| ty::mk_rptr(tcx, r, tmt))
      }
      ast::ty_tup(fields) => {
        let flds = vec::map(fields, |t| ast_ty_to_ty(self, rscope, *t));
        ty::mk_tup(tcx, flds)
      }
      ast::ty_rec(fields) => {
        let flds = do fields.map |f| {
            let tm = ast_mt_to_mt(self, rscope, f.node.mt);
            {ident: f.node.ident, mt: tm}
        };
        ty::mk_rec(tcx, flds)
      }
      ast::ty_fn(proto, purity, ast_bounds, decl) => {
        let bounds = collect::compute_bounds(self.ccx(), ast_bounds);
        let fn_decl = ty_of_fn_decl(self, rscope, proto, purity,
                                    bounds, decl, None,
                                    ast_ty.span);
        ty::mk_fn(tcx, fn_decl)
      }
      ast::ty_path(path, id) => {
        let a_def = match tcx.def_map.find(id) {
          None => tcx.sess.span_fatal(
              ast_ty.span, fmt!("unbound path %s",
                                path_to_str(path, tcx.sess.intr()))),
          Some(d) => d
        };
        match a_def {
          ast::def_ty(did) | ast::def_class(did) => {
            ast_path_to_ty(self, rscope, did, path, id).ty
          }
          ast::def_prim_ty(nty) => {
            match nty {
              ast::ty_bool => {
                check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                ty::mk_bool(tcx)
              }
              ast::ty_int(it) => {
                check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                ty::mk_mach_int(tcx, it)
              }
              ast::ty_uint(uit) => {
                check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                ty::mk_mach_uint(tcx, uit)
              }
              ast::ty_float(ft) => {
                check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                ty::mk_mach_float(tcx, ft)
              }
              ast::ty_str => {
                tcx.sess.span_err(ast_ty.span,
                                  ~"bare `str` is not a type");
                // return /something/ so they can at least get more errors
                ty::mk_estr(tcx, ty::vstore_uniq)
              }
            }
          }
          ast::def_ty_param(id, n) => {
            check_path_args(tcx, path, NO_TPS | NO_REGIONS);
            ty::mk_param(tcx, n, id)
          }
          ast::def_self(_) => {
            // n.b.: resolve guarantees that the self type only appears in a
            // trait, which we rely upon in various places when creating
            // substs
            check_path_args(tcx, path, NO_TPS | NO_REGIONS);
            ty::mk_self(tcx)
          }
          _ => {
            tcx.sess.span_fatal(ast_ty.span,
                                ~"found type name used as a variable");
          }
        }
      }
      ast::ty_fixed_length(a_t, Some(u)) => {
        mk_maybe_vstore(self, rscope, {ty: a_t, mutbl: ast::m_imm},
                        ty::vstore_fixed(u),
                        ast_ty.span,
                        |ty| {
                            tcx.sess.span_err(
                                a_t.span,
                                fmt!("bound not allowed on a %s",
                                     ty::ty_sort_str(tcx, ty.ty)));
                            ty.ty
                        })
      }
      ast::ty_fixed_length(_, None) => {
        tcx.sess.span_bug(
            ast_ty.span,
            ~"implied fixed length for bound");
      }
      ast::ty_infer => {
        // ty_infer should only appear as the type of arguments or return
        // values in a fn_expr, or as the type of local variables.  Both of
        // these cases are handled specially and should not descend into this
        // routine.
        self.tcx().sess.span_bug(
            ast_ty.span,
            ~"found `ty_infer` in unexpected place");
      }
      ast::ty_mac(_) => {
        tcx.sess.span_bug(ast_ty.span,
                          ~"found `ty_mac` in unexpected place");
      }
    };

    tcx.ast_ty_to_ty_cache.insert(ast_ty, ty::atttce_resolved(typ));
    return typ;
}

fn ty_of_arg<AC: ast_conv, RS: region_scope Copy Owned>(
    self: AC, rscope: RS, a: ast::arg,
    expected_ty: Option<ty::arg>) -> ty::arg {

    let ty = match a.ty.node {
      ast::ty_infer if expected_ty.is_some() => expected_ty.get().ty,
      ast::ty_infer => self.ty_infer(a.ty.span),
      _ => ast_ty_to_ty(self, rscope, a.ty)
    };

    let mode = {
        match a.mode {
          ast::infer(_) if expected_ty.is_some() => {
            result::get(&ty::unify_mode(
                self.tcx(),
                ty::expected_found {expected: expected_ty.get().mode,
                                    found: a.mode}))
          }
          ast::infer(_) => {
            match ty::get(ty).sty {
              // If the type is not specified, then this must be a fn expr.
              // Leave the mode as infer(_), it will get inferred based
              // on constraints elsewhere.
              ty::ty_infer(_) => a.mode,

              // If the type is known, then use the default for that type.
              // Here we unify m and the default.  This should update the
              // tables in tcx but should never fail, because nothing else
              // will have been unified with m yet:
              _ => {
                let m1 = ast::expl(ty::default_arg_mode_for_ty(self.tcx(),
                                                               ty));
                result::get(&ty::unify_mode(
                    self.tcx(),
                    ty::expected_found {expected: m1,
                                        found: a.mode}))
              }
            }
          }
          ast::expl(_) => a.mode
        }
    };

    {mode: mode, ty: ty}
}

fn ast_proto_to_proto<AC: ast_conv, RS: region_scope Copy Owned>(
    self: AC, rscope: RS, span: span, ast_proto: ast::proto) -> ty::fn_proto {
    match ast_proto {
        ast::proto_bare =>
            ty::proto_bare,
        ast::proto_uniq =>
            ty::proto_vstore(ty::vstore_uniq),
        ast::proto_box =>
            ty::proto_vstore(ty::vstore_box),
        ast::proto_block => {
            let result = rscope.anon_region(span);
            let region = get_region_reporting_err(self.tcx(), span, result);
            ty::proto_vstore(ty::vstore_slice(region))
        }
    }
}

type expected_tys = Option<{inputs: ~[ty::arg],
                            output: ty::t}>;

fn ty_of_fn_decl<AC: ast_conv, RS: region_scope Copy Owned>(
    self: AC, rscope: RS,
    ast_proto: ast::proto,
    purity: ast::purity,
    bounds: @~[ty::param_bound],
    decl: ast::fn_decl,
    expected_tys: expected_tys,
    span: span) -> ty::FnTy {

    debug!("ty_of_fn_decl");
    do indent {
        // new region names that appear inside of the fn decl are bound to
        // that function type
        let rb = in_binding_rscope(rscope);

        let input_tys = do decl.inputs.mapi |i, a| {
            let expected_arg_ty = do expected_tys.chain_ref |e| {
                // no guarantee that the correct number of expected args
                // were supplied
                if i < e.inputs.len() {Some(e.inputs[i])} else {None}
            };
            ty_of_arg(self, rb, *a, expected_arg_ty)
        };

        let expected_ret_ty = expected_tys.map(|e| e.output);
        let output_ty = match decl.output.node {
          ast::ty_infer if expected_ret_ty.is_some() => expected_ret_ty.get(),
          ast::ty_infer => self.ty_infer(decl.output.span),
          _ => ast_ty_to_ty(self, rb, decl.output)
        };

        let proto = ast_proto_to_proto(self, rscope, span, ast_proto);

        FnTyBase {
            meta: FnMeta {purity: purity,
                          proto: proto,
                          bounds: bounds,
                          ret_style: decl.cf},
            sig: FnSig {inputs: input_tys,
                        output: output_ty}
        }
    }
}


