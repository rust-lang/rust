// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Conversion from AST representation of types to the ty.rs
 * representation.  The main routine here is `ast_ty_to_ty()`: each use
 * is parameterized by an instance of `AstConv` and a `region_scope`.
 *
 * The parameterization of `ast_ty_to_ty()` is because it behaves
 * somewhat differently during the collect and check phases, particularly
 * with respect to looking up the types of top-level items.  In the
 * collect phase, the crate context is used as the `AstConv` instance;
 * in this phase, the `get_item_ty()` function triggers a recursive call
 * to `ty_of_item()` (note that `ast_ty_to_ty()` will detect recursive
 * types and report an error).  In the check phase, when the @FnCtxt is
 * used as the `AstConv`, `get_item_ty()` just looks up the item type in
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
 * Unlike the `AstConv` trait, the region scope can change as we descend
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

use core::prelude::*;

use middle::ty::{arg, field, substs};
use middle::ty::{ty_param_substs_and_ty};
use middle::ty;
use middle::typeck::rscope::{in_anon_rscope, in_binding_rscope};
use middle::typeck::rscope::{region_scope, type_rscope};
use middle::typeck::{CrateCtxt, write_substs_to_tcx, write_ty_to_tcx};

use core::result;
use core::vec;
use syntax::ast;
use syntax::codemap::span;
use syntax::print::pprust::path_to_str;
use util::common::indenter;

pub trait AstConv {
    fn tcx(&self) -> ty::ctxt;
    fn get_item_ty(&self, id: ast::def_id) -> ty::ty_param_bounds_and_ty;

    // what type should we use when a type is omitted?
    fn ty_infer(&self, span: span) -> ty::t;
}

pub fn get_region_reporting_err(tcx: ty::ctxt,
                                span: span,
                                res: Result<ty::Region, ~str>)
                             -> ty::Region {

    match res {
      result::Ok(r) => r,
      result::Err(ref e) => {
        tcx.sess.span_err(span, (/*bad*/copy *e));
        ty::re_static
      }
    }
}

pub fn ast_region_to_region<AC:AstConv,RS:region_scope + Copy + Durable>(
        self: &AC,
        rscope: &RS,
        span: span,
        a_r: @ast::region)
     -> ty::Region {
    let res = match a_r.node {
        ast::re_static => Ok(ty::re_static),
        ast::re_anon => rscope.anon_region(span),
        ast::re_self => rscope.self_region(span),
        ast::re_named(id) => rscope.named_region(span, id)
    };

    get_region_reporting_err(self.tcx(), span, res)
}

pub fn ast_path_to_substs_and_ty<AC:AstConv,RS:region_scope + Copy + Durable>(
        self: &AC,
        rscope: &RS,
        did: ast::def_id,
        path: @ast::path)
     -> ty_param_substs_and_ty {
    let tcx = self.tcx();
    let ty::ty_param_bounds_and_ty {
        bounds: decl_bounds,
        region_param: decl_rp,
        ty: decl_ty
    } = self.get_item_ty(did);

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

    let substs = substs {self_r:self_r, self_ty:None, tps:tps};
    let ty = ty::subst(tcx, &substs, decl_ty);

    ty_param_substs_and_ty { substs: substs, ty: ty }
}

pub fn ast_path_to_ty<AC:AstConv,RS:region_scope + Copy + Durable>(
        self: &AC,
        rscope: &RS,
        did: ast::def_id,
        path: @ast::path,
        path_id: ast::node_id)
     -> ty_param_substs_and_ty {
    // Look up the polytype of the item and then substitute the provided types
    // for any type/region parameters.
    let tcx = self.tcx();
    let ty::ty_param_substs_and_ty {
        substs: substs,
        ty: ty
    } = ast_path_to_substs_and_ty(self, rscope, did, path);
    write_ty_to_tcx(tcx, path_id, ty);
    write_substs_to_tcx(tcx, path_id, /*bad*/copy substs.tps);

    ty_param_substs_and_ty { substs: substs, ty: ty }
}

pub const NO_REGIONS: uint = 1;
pub const NO_TPS: uint = 2;

// Parses the programmer's textual representation of a type into our
// internal notion of a type. `getter` is a function that returns the type
// corresponding to a definition ID:
pub fn ast_ty_to_ty<AC:AstConv, RS:region_scope + Copy + Durable>(
    self: &AC, rscope: &RS, &&ast_ty: @ast::Ty) -> ty::t {

    fn ast_mt_to_mt<AC:AstConv, RS:region_scope + Copy + Durable>(
        self: &AC, rscope: &RS, mt: ast::mt) -> ty::mt {

        ty::mt {ty: ast_ty_to_ty(self, rscope, mt.ty), mutbl: mt.mutbl}
    }

    // Handle @, ~, and & being able to mean estrs and evecs.
    // If a_seq_ty is a str or a vec, make it an estr/evec.
    // Also handle function sigils and first-class trait types.
    fn mk_pointer<AC:AstConv,RS:region_scope + Copy + Durable>(
        self: &AC,
        rscope: &RS,
        a_seq_ty: ast::mt,
        vst: ty::vstore,
        constr: fn(ty::mt) -> ty::t) -> ty::t
    {
        let tcx = self.tcx();

        match a_seq_ty.ty.node {
            ast::ty_vec(mt) => {
                let mut mt = ast_mt_to_mt(self, rscope, mt);
                if a_seq_ty.mutbl == ast::m_mutbl ||
                        a_seq_ty.mutbl == ast::m_const {
                    mt = ty::mt { ty: mt.ty, mutbl: a_seq_ty.mutbl };
                }
                return ty::mk_evec(tcx, mt, vst);
            }
            ast::ty_path(path, id) if a_seq_ty.mutbl == ast::m_imm => {
                match tcx.def_map.find(&id) {
                    Some(ast::def_prim_ty(ast::ty_str)) => {
                        check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                        return ty::mk_estr(tcx, vst);
                    }
                    Some(ast::def_ty(type_def_id)) => {
                        let result = ast_path_to_substs_and_ty(
                            self, rscope,
                            type_def_id, path);
                        match ty::get(result.ty).sty {
                            ty::ty_trait(trait_def_id, ref substs, _) => {
                                match vst {
                                    ty::vstore_box | ty::vstore_slice(*) |
                                    ty::vstore_uniq => {}
                                    _ => {
                                        tcx.sess.span_err(
                                            path.span,
                                            ~"@trait, ~trait or &trait \
                                              are the only supported \
                                              forms of casting-to-\
                                              trait");
                                    }
                                }
                                return ty::mk_trait(tcx, trait_def_id,
                                                    /*bad*/copy *substs, vst);

                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
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

    match tcx.ast_ty_to_ty_cache.find(&ast_ty.id) {
      Some(ty::atttce_resolved(ty)) => return ty,
      Some(ty::atttce_unresolved) => {
        tcx.sess.span_fatal(ast_ty.span, ~"illegal recursive type; \
                                          insert an enum in the cycle, \
                                          if this is desired");
      }
      None => { /* go on */ }
    }

    tcx.ast_ty_to_ty_cache.insert(ast_ty.id, ty::atttce_unresolved);
    let typ = match /*bad*/copy ast_ty.node {
      ast::ty_nil => ty::mk_nil(tcx),
      ast::ty_bot => ty::mk_bot(tcx),
      ast::ty_box(mt) => {
        mk_pointer(self, rscope, mt, ty::vstore_box,
                   |tmt| ty::mk_box(tcx, tmt))
      }
      ast::ty_uniq(mt) => {
        mk_pointer(self, rscope, mt, ty::vstore_uniq,
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
        let anon_rscope = in_anon_rscope(rscope, r);
        mk_pointer(self, &anon_rscope, mt, ty::vstore_slice(r),
                   |tmt| ty::mk_rptr(tcx, r, tmt))
      }
      ast::ty_tup(fields) => {
        let flds = vec::map(fields, |t| ast_ty_to_ty(self, rscope, *t));
        ty::mk_tup(tcx, flds)
      }
      ast::ty_bare_fn(ref bf) => {
          ty::mk_bare_fn(tcx, ty_of_bare_fn(self, rscope, bf.purity,
                                            bf.abi, &bf.decl))
      }
      ast::ty_closure(ref f) => {
          let fn_decl = ty_of_closure(self, rscope, f.sigil,
                                      f.purity, f.onceness,
                                      f.region, &f.decl, None,
                                      ast_ty.span);
          ty::mk_closure(tcx, fn_decl)
      }
      ast::ty_path(path, id) => {
        let a_def = match tcx.def_map.find(&id) {
          None => tcx.sess.span_fatal(
              ast_ty.span, fmt!("unbound path %s",
                                path_to_str(path, tcx.sess.intr()))),
          Some(d) => d
        };
        match a_def {
          ast::def_ty(did) | ast::def_struct(did) => {
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
          ast::def_self_ty(_) => {
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
      ast::ty_fixed_length_vec(a_mt, u) => {
        ty::mk_evec(tcx, ast_mt_to_mt(self, rscope, a_mt),
                    ty::vstore_fixed(u))
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

    tcx.ast_ty_to_ty_cache.insert(ast_ty.id, ty::atttce_resolved(typ));
    return typ;
}

pub fn ty_of_arg<AC:AstConv,RS:region_scope + Copy + Durable>(
        self: &AC,
        rscope: &RS,
        a: ast::arg,
        expected_ty: Option<ty::arg>)
     -> ty::arg {
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

    arg {mode: mode, ty: ty}
}

pub fn ty_of_bare_fn<AC:AstConv,RS:region_scope + Copy + Durable>(
        self: &AC,
        rscope: &RS,
        purity: ast::purity,
        abi: ast::Abi,
        decl: &ast::fn_decl)
     -> ty::BareFnTy {
    debug!("ty_of_fn_decl");

    // new region names that appear inside of the fn decl are bound to
    // that function type
    let rb = in_binding_rscope(rscope);

    let input_tys = decl.inputs.map(|a| ty_of_arg(self, &rb, *a, None));
    let output_ty = match decl.output.node {
        ast::ty_infer => self.ty_infer(decl.output.span),
        _ => ast_ty_to_ty(self, &rb, decl.output)
    };

    ty::BareFnTy {
        purity: purity,
        abi: abi,
        sig: ty::FnSig {inputs: input_tys, output: output_ty}
    }
}

pub fn ty_of_closure<AC:AstConv,RS:region_scope + Copy + Durable>(
        self: &AC,
        rscope: &RS,
        sigil: ast::Sigil,
        purity: ast::purity,
        onceness: ast::Onceness,
        opt_region: Option<@ast::region>,
        decl: &ast::fn_decl,
        expected_tys: Option<ty::FnSig>,
        span: span)
     -> ty::ClosureTy {
    debug!("ty_of_fn_decl");
    let _i = indenter();

    // resolve the function bound region in the original region
    // scope `rscope`, not the scope of the function parameters
    let bound_region = match opt_region {
        Some(region) => {
            ast_region_to_region(self, rscope, span, region)
        }
        None => {
            match sigil {
                ast::OwnedSigil | ast::ManagedSigil => {
                    // @fn(), ~fn() default to static as the bound
                    // on their upvars:
                    ty::re_static
                }
                ast::BorrowedSigil => {
                    // &fn() defaults to an anonymous region:
                    let r_result = rscope.anon_region(span);
                    get_region_reporting_err(self.tcx(), span, r_result)
                }
            }
        }
    };

    // new region names that appear inside of the fn decl are bound to
    // that function type
    let rb = in_binding_rscope(rscope);

    let input_tys = do decl.inputs.mapi |i, a| {
        let expected_arg_ty = do expected_tys.chain_ref |e| {
            // no guarantee that the correct number of expected args
            // were supplied
            if i < e.inputs.len() {Some(e.inputs[i])} else {None}
        };
        ty_of_arg(self, &rb, *a, expected_arg_ty)
    };

    let expected_ret_ty = expected_tys.map(|e| e.output);
    let output_ty = match decl.output.node {
        ast::ty_infer if expected_ret_ty.is_some() => expected_ret_ty.get(),
        ast::ty_infer => self.ty_infer(decl.output.span),
        _ => ast_ty_to_ty(self, &rb, decl.output)
    };

    ty::ClosureTy {
        purity: purity,
        sigil: sigil,
        onceness: onceness,
        region: bound_region,
        sig: ty::FnSig {inputs: input_tys,
                        output: output_ty}
    }
}
