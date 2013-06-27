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
 * somewhat differently during the collect and check phases,
 * particularly with respect to looking up the types of top-level
 * items.  In the collect phase, the crate context is used as the
 * `AstConv` instance; in this phase, the `get_item_ty()` function
 * triggers a recursive call to `ty_of_item()`  (note that
 * `ast_ty_to_ty()` will detect recursive types and report an error).
 * In the check phase, when the @FnCtxt is used as the `AstConv`,
 * `get_item_ty()` just looks up the item type in `tcx.tcache`.
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
 *   type foo = { x: &a.int, y: &fn(&a.int) }
 *
 * The type of `x` is an error because there is no region `a` in scope.
 * In the type of `y`, however, region `a` is considered a bound region
 * as it does not already appear in scope.
 *
 * Case (b) says that if you have a type:
 *   type foo<'self> = ...;
 *   type bar = fn(&foo, &a.foo)
 * The fully expanded version of type bar is:
 *   type bar = fn(&'foo &, &a.foo<'a>)
 * Note that the self region for the `foo` defaulted to `&` in the first
 * case but `&a` in the second.  Basically, defaults that appear inside
 * an rptr (`&r.T`) use the region `r` that appears in the rptr.
 */

use core::prelude::*;

use middle::const_eval;
use middle::ty::{substs};
use middle::ty::{ty_param_substs_and_ty};
use middle::ty;
use middle::typeck::rscope::in_binding_rscope;
use middle::typeck::rscope::{region_scope, RegionError};
use middle::typeck::rscope::RegionParamNames;
use middle::typeck::lookup_def_tcx;

use core::result;
use core::vec;
use syntax::abi::AbiSet;
use syntax::{ast, ast_util};
use syntax::codemap::span;
use syntax::opt_vec::OptVec;
use syntax::opt_vec;
use syntax::print::pprust::{lifetime_to_str, path_to_str};
use syntax::parse::token::special_idents;
use util::common::indenter;

pub trait AstConv {
    fn tcx(&self) -> ty::ctxt;
    fn get_item_ty(&self, id: ast::def_id) -> ty::ty_param_bounds_and_ty;
    fn get_trait_def(&self, id: ast::def_id) -> @ty::TraitDef;

    // what type should we use when a type is omitted?
    fn ty_infer(&self, span: span) -> ty::t;
}

pub fn get_region_reporting_err(
    tcx: ty::ctxt,
    span: span,
    a_r: Option<@ast::Lifetime>,
    res: Result<ty::Region, RegionError>) -> ty::Region
{
    match res {
        result::Ok(r) => r,
        result::Err(ref e) => {
            let descr = match a_r {
                None => ~"anonymous lifetime",
                Some(a) => fmt!("lifetime %s",
                                lifetime_to_str(a, tcx.sess.intr()))
            };
            tcx.sess.span_err(
                span,
                fmt!("Illegal %s: %s",
                     descr, e.msg));
            e.replacement
        }
    }
}

pub fn ast_region_to_region<AC:AstConv,RS:region_scope + Copy + 'static>(
    this: &AC,
    rscope: &RS,
    default_span: span,
    opt_lifetime: Option<@ast::Lifetime>) -> ty::Region
{
    let (span, res) = match opt_lifetime {
        None => {
            (default_span, rscope.anon_region(default_span))
        }
        Some(ref lifetime) if lifetime.ident == special_idents::statik => {
            (lifetime.span, Ok(ty::re_static))
        }
        Some(ref lifetime) if lifetime.ident == special_idents::self_ => {
            (lifetime.span, rscope.self_region(lifetime.span))
        }
        Some(ref lifetime) => {
            (lifetime.span, rscope.named_region(lifetime.span,
                                                lifetime.ident))
        }
    };

    get_region_reporting_err(this.tcx(), span, opt_lifetime, res)
}

fn ast_path_substs<AC:AstConv,RS:region_scope + Copy + 'static>(
    this: &AC,
    rscope: &RS,
    def_id: ast::def_id,
    decl_generics: &ty::Generics,
    self_ty: Option<ty::t>,
    path: @ast::Path) -> ty::substs
{
    /*!
     *
     * Given a path `path` that refers to an item `I` with the
     * declared generics `decl_generics`, returns an appropriate
     * set of substitutions for this particular reference to `I`.
     */

    let tcx = this.tcx();

    // If the type is parameterized by the this region, then replace this
    // region with the current anon region binding (in other words,
    // whatever & would get replaced with).
    let self_r = match (&decl_generics.region_param, &path.rp) {
      (&None, &None) => {
        None
      }
      (&None, &Some(_)) => {
        tcx.sess.span_err(
            path.span,
            fmt!("no region bound is allowed on `%s`, \
                  which is not declared as containing region pointers",
                 ty::item_path_str(tcx, def_id)));
        None
      }
      (&Some(_), &None) => {
        let res = rscope.anon_region(path.span);
        let r = get_region_reporting_err(this.tcx(), path.span, None, res);
        Some(r)
      }
      (&Some(_), &Some(_)) => {
        Some(ast_region_to_region(this, rscope, path.span, path.rp))
      }
    };

    // Convert the type parameters supplied by the user.
    if !vec::same_length(*decl_generics.type_param_defs, path.types) {
        this.tcx().sess.span_fatal(
            path.span,
            fmt!("wrong number of type arguments: expected %u but found %u",
                 decl_generics.type_param_defs.len(), path.types.len()));
    }
    let tps = path.types.map(|a_t| ast_ty_to_ty(this, rscope, *a_t));

    substs {self_r:self_r, self_ty:self_ty, tps:tps}
}

pub fn ast_path_to_substs_and_ty<AC:AstConv,RS:region_scope + Copy + 'static>(
    this: &AC,
    rscope: &RS,
    did: ast::def_id,
    path: @ast::Path) -> ty_param_substs_and_ty
{
    let tcx = this.tcx();
    let ty::ty_param_bounds_and_ty {
        generics: generics,
        ty: decl_ty
    } = this.get_item_ty(did);

    let substs = ast_path_substs(this, rscope, did, &generics, None, path);
    let ty = ty::subst(tcx, &substs, decl_ty);
    ty_param_substs_and_ty { substs: substs, ty: ty }
}

pub fn ast_path_to_trait_ref<AC:AstConv,RS:region_scope + Copy + 'static>(
    this: &AC,
    rscope: &RS,
    trait_def_id: ast::def_id,
    self_ty: Option<ty::t>,
    path: @ast::Path) -> @ty::TraitRef
{
    let trait_def =
        this.get_trait_def(trait_def_id);
    let substs =
        ast_path_substs(
            this,
            rscope,
            trait_def.trait_ref.def_id,
            &trait_def.generics,
            self_ty,
            path);
    let trait_ref =
        @ty::TraitRef {def_id: trait_def_id,
                       substs: substs};
    return trait_ref;
}

pub fn ast_path_to_ty<AC:AstConv,RS:region_scope + Copy + 'static>(
        this: &AC,
        rscope: &RS,
        did: ast::def_id,
        path: @ast::Path)
     -> ty_param_substs_and_ty
{
    // Look up the polytype of the item and then substitute the provided types
    // for any type/region parameters.
    let ty::ty_param_substs_and_ty {
        substs: substs,
        ty: ty
    } = ast_path_to_substs_and_ty(this, rscope, did, path);
    ty_param_substs_and_ty { substs: substs, ty: ty }
}

pub static NO_REGIONS: uint = 1;
pub static NO_TPS: uint = 2;

// Parses the programmer's textual representation of a type into our
// internal notion of a type. `getter` is a function that returns the type
// corresponding to a definition ID:
pub fn ast_ty_to_ty<AC:AstConv, RS:region_scope + Copy + 'static>(
    this: &AC, rscope: &RS, ast_ty: @ast::Ty) -> ty::t {

    fn ast_mt_to_mt<AC:AstConv, RS:region_scope + Copy + 'static>(
        this: &AC, rscope: &RS, mt: &ast::mt) -> ty::mt {

        ty::mt {ty: ast_ty_to_ty(this, rscope, mt.ty), mutbl: mt.mutbl}
    }

    // Handle @, ~, and & being able to mean estrs and evecs.
    // If a_seq_ty is a str or a vec, make it an estr/evec.
    // Also handle first-class trait types.
    fn mk_pointer<AC:AstConv,RS:region_scope + Copy + 'static>(
        this: &AC,
        rscope: &RS,
        a_seq_ty: &ast::mt,
        vst: ty::vstore,
        constr: &fn(ty::mt) -> ty::t) -> ty::t
    {
        let tcx = this.tcx();

        match a_seq_ty.ty.node {
            ast::ty_vec(ref mt) => {
                let mut mt = ast_mt_to_mt(this, rscope, mt);
                if a_seq_ty.mutbl == ast::m_mutbl ||
                        a_seq_ty.mutbl == ast::m_const {
                    mt = ty::mt { ty: mt.ty, mutbl: a_seq_ty.mutbl };
                }
                return ty::mk_evec(tcx, mt, vst);
            }
            ast::ty_path(path, bounds, id) => {
                // Note that the "bounds must be empty if path is not a trait"
                // restriction is enforced in the below case for ty_path, which
                // will run after this as long as the path isn't a trait.
                match tcx.def_map.find(&id) {
                    Some(&ast::def_prim_ty(ast::ty_str)) if a_seq_ty.mutbl == ast::m_imm => {
                        check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                        return ty::mk_estr(tcx, vst);
                    }
                    Some(&ast::def_trait(trait_def_id)) => {
                        let result = ast_path_to_trait_ref(
                            this, rscope, trait_def_id, None, path);
                        let trait_store = match vst {
                            ty::vstore_box => ty::BoxTraitStore,
                            ty::vstore_uniq => ty::UniqTraitStore,
                            ty::vstore_slice(r) => {
                                ty::RegionTraitStore(r)
                            }
                            ty::vstore_fixed(*) => {
                                tcx.sess.span_err(
                                    path.span,
                                    "@trait, ~trait or &trait are the only supported \
                                     forms of casting-to-trait");
                                ty::BoxTraitStore
                            }
                        };
                        let bounds = conv_builtin_bounds(this.tcx(), bounds, trait_store);
                        return ty::mk_trait(tcx,
                                            result.def_id,
                                            copy result.substs,
                                            trait_store,
                                            a_seq_ty.mutbl,
                                            bounds);
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        let seq_ty = ast_mt_to_mt(this, rscope, a_seq_ty);
        return constr(seq_ty);
    }

    fn check_path_args(tcx: ty::ctxt,
                       path: @ast::Path,
                       flags: uint) {
        if (flags & NO_TPS) != 0u {
            if path.types.len() > 0u {
                tcx.sess.span_err(
                    path.span,
                    "type parameters are not allowed on this type");
            }
        }

        if (flags & NO_REGIONS) != 0u {
            if path.rp.is_some() {
                tcx.sess.span_err(
                    path.span,
                    "region parameters are not allowed on this type");
            }
        }
    }

    let tcx = this.tcx();

    match tcx.ast_ty_to_ty_cache.find(&ast_ty.id) {
      Some(&ty::atttce_resolved(ty)) => return ty,
      Some(&ty::atttce_unresolved) => {
        tcx.sess.span_fatal(ast_ty.span, "illegal recursive type; \
                                          insert an enum in the cycle, if this is desired");
      }
      None => { /* go on */ }
    }

    tcx.ast_ty_to_ty_cache.insert(ast_ty.id, ty::atttce_unresolved);
    let typ = match ast_ty.node {
      ast::ty_nil => ty::mk_nil(),
      ast::ty_bot => ty::mk_bot(),
      ast::ty_box(ref mt) => {
        mk_pointer(this, rscope, mt, ty::vstore_box,
                   |tmt| ty::mk_box(tcx, tmt))
      }
      ast::ty_uniq(ref mt) => {
        mk_pointer(this, rscope, mt, ty::vstore_uniq,
                   |tmt| ty::mk_uniq(tcx, tmt))
      }
      ast::ty_vec(ref mt) => {
        tcx.sess.span_err(ast_ty.span, "bare `[]` is not a type");
        // return /something/ so they can at least get more errors
        ty::mk_evec(tcx, ast_mt_to_mt(this, rscope, mt), ty::vstore_uniq)
      }
      ast::ty_ptr(ref mt) => {
        ty::mk_ptr(tcx, ast_mt_to_mt(this, rscope, mt))
      }
      ast::ty_rptr(region, ref mt) => {
        let r = ast_region_to_region(this, rscope, ast_ty.span, region);
        mk_pointer(this, rscope, mt, ty::vstore_slice(r),
                   |tmt| ty::mk_rptr(tcx, r, tmt))
      }
      ast::ty_tup(ref fields) => {
        let flds = fields.map(|t| ast_ty_to_ty(this, rscope, *t));
        ty::mk_tup(tcx, flds)
      }
      ast::ty_bare_fn(ref bf) => {
          ty::mk_bare_fn(tcx, ty_of_bare_fn(this, rscope, bf.purity,
                                            bf.abis, &bf.lifetimes, &bf.decl))
      }
      ast::ty_closure(ref f) => {
          let bounds = conv_builtin_bounds(this.tcx(), &f.bounds, match f.sigil {
              // Use corresponding trait store to figure out default bounds
              // if none were specified.
              ast::BorrowedSigil => ty::RegionTraitStore(ty::re_empty), // dummy region
              ast::OwnedSigil    => ty::UniqTraitStore,
              ast::ManagedSigil  => ty::BoxTraitStore,
          });
          let fn_decl = ty_of_closure(this,
                                      rscope,
                                      f.sigil,
                                      f.purity,
                                      f.onceness,
                                      bounds,
                                      f.region,
                                      &f.decl,
                                      None,
                                      &f.lifetimes,
                                      ast_ty.span);
          ty::mk_closure(tcx, fn_decl)
      }
      ast::ty_path(path, bounds, id) => {
        let a_def = match tcx.def_map.find(&id) {
          None => tcx.sess.span_fatal(
              ast_ty.span, fmt!("unbound path %s",
                                path_to_str(path, tcx.sess.intr()))),
          Some(&d) => d
        };
        // Kind bounds on path types are only supported for traits.
        match a_def {
            // But don't emit the error if the user meant to do a trait anyway.
            ast::def_trait(*) => { },
            _ if bounds.is_some() =>
                tcx.sess.span_err(ast_ty.span,
                    "kind bounds can only be used on trait types"),
            _ => { },
        }
        match a_def {
          ast::def_trait(_) => {
              let path_str = path_to_str(path, tcx.sess.intr());
              tcx.sess.span_err(
                  ast_ty.span,
                  fmt!("reference to trait `%s` where a type is expected; \
                        try `@%s`, `~%s`, or `&%s`",
                       path_str, path_str, path_str, path_str));
              ty::mk_err()
          }
          ast::def_ty(did) | ast::def_struct(did) => {
            ast_path_to_ty(this, rscope, did, path).ty
          }
          ast::def_prim_ty(nty) => {
            match nty {
              ast::ty_bool => {
                check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                ty::mk_bool()
              }
              ast::ty_int(it) => {
                check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                ty::mk_mach_int(it)
              }
              ast::ty_uint(uit) => {
                check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                ty::mk_mach_uint(uit)
              }
              ast::ty_float(ft) => {
                check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                ty::mk_mach_float(ft)
              }
              ast::ty_str => {
                tcx.sess.span_err(ast_ty.span,
                                  "bare `str` is not a type");
                // return /something/ so they can at least get more errors
                ty::mk_estr(tcx, ty::vstore_uniq)
              }
            }
          }
          ast::def_ty_param(id, n) => {
            check_path_args(tcx, path, NO_TPS | NO_REGIONS);
            ty::mk_param(tcx, n, id)
          }
          ast::def_self_ty(id) => {
            // n.b.: resolve guarantees that the this type only appears in a
            // trait, which we rely upon in various places when creating
            // substs
            check_path_args(tcx, path, NO_TPS | NO_REGIONS);
            let did = ast_util::local_def(id);
            ty::mk_self(tcx, did)
          }
          _ => {
            tcx.sess.span_fatal(ast_ty.span,
                                fmt!("found value name used as a type: %?", a_def));
          }
        }
      }
      ast::ty_fixed_length_vec(ref a_mt, e) => {
        match const_eval::eval_const_expr_partial(tcx, e) {
          Ok(ref r) => {
            match *r {
              const_eval::const_int(i) =>
                ty::mk_evec(tcx, ast_mt_to_mt(this, rscope, a_mt),
                            ty::vstore_fixed(i as uint)),
              const_eval::const_uint(i) =>
                ty::mk_evec(tcx, ast_mt_to_mt(this, rscope, a_mt),
                            ty::vstore_fixed(i as uint)),
              _ => {
                tcx.sess.span_fatal(
                    ast_ty.span, "expected constant expr for vector length");
              }
            }
          }
          Err(ref r) => {
            tcx.sess.span_fatal(
                ast_ty.span,
                fmt!("expected constant expr for vector length: %s",
                     *r));
          }
        }
      }
      ast::ty_infer => {
        // ty_infer should only appear as the type of arguments or return
        // values in a fn_expr, or as the type of local variables.  Both of
        // these cases are handled specially and should not descend into this
        // routine.
        this.tcx().sess.span_bug(
            ast_ty.span,
            "found `ty_infer` in unexpected place");
      }
      ast::ty_mac(_) => {
        tcx.sess.span_bug(ast_ty.span,
                          "found `ty_mac` in unexpected place");
      }
    };

    tcx.ast_ty_to_ty_cache.insert(ast_ty.id, ty::atttce_resolved(typ));
    return typ;
}

pub fn ty_of_arg<AC:AstConv,
                 RS:region_scope + Copy + 'static>(
                 this: &AC,
                 rscope: &RS,
                 a: ast::arg,
                 expected_ty: Option<ty::t>)
                 -> ty::t {
    match a.ty.node {
        ast::ty_infer if expected_ty.is_some() => expected_ty.get(),
        ast::ty_infer => this.ty_infer(a.ty.span),
        _ => ast_ty_to_ty(this, rscope, a.ty),
    }
}

pub fn bound_lifetimes<AC:AstConv>(
    this: &AC,
    ast_lifetimes: &OptVec<ast::Lifetime>) -> OptVec<ast::ident>
{
    /*!
     *
     * Converts a list of lifetimes into a list of bound identifier
     * names.  Does not permit special names like 'static or 'this to
     * be bound.  Note that this function is for use in closures,
     * methods, and fn definitions.  It is legal to bind 'this in a
     * type.  Eventually this distinction should go away and the same
     * rules should apply everywhere ('this would not be a special name
     * at that point).
     */

    let special_idents = [special_idents::statik, special_idents::self_];
    let mut bound_lifetime_names = opt_vec::Empty;
    ast_lifetimes.map_to_vec(|ast_lifetime| {
        if special_idents.iter().any_(|&i| i == ast_lifetime.ident) {
            this.tcx().sess.span_err(
                ast_lifetime.span,
                fmt!("illegal lifetime parameter name: `%s`",
                     lifetime_to_str(ast_lifetime, this.tcx().sess.intr())));
        } else {
            bound_lifetime_names.push(ast_lifetime.ident);
        }
    });
    bound_lifetime_names
}

struct SelfInfo {
    untransformed_self_ty: ty::t,
    explicit_self: ast::explicit_self
}

pub fn ty_of_method<AC:AstConv,RS:region_scope + Copy + 'static>(
    this: &AC,
    rscope: &RS,
    purity: ast::purity,
    lifetimes: &OptVec<ast::Lifetime>,
    untransformed_self_ty: ty::t,
    explicit_self: ast::explicit_self,
    decl: &ast::fn_decl) -> (Option<ty::t>, ty::BareFnTy)
{
    let self_info = SelfInfo {
        untransformed_self_ty: untransformed_self_ty,
        explicit_self: explicit_self
    };
    let (a, b) = ty_of_method_or_bare_fn(
        this, rscope, purity, AbiSet::Rust(), lifetimes, Some(&self_info), decl);
    (a.get(), b)
}

pub fn ty_of_bare_fn<AC:AstConv,RS:region_scope + Copy + 'static>(
    this: &AC,
    rscope: &RS,
    purity: ast::purity,
    abi: AbiSet,
    lifetimes: &OptVec<ast::Lifetime>,
    decl: &ast::fn_decl) -> ty::BareFnTy
{
    let (_, b) = ty_of_method_or_bare_fn(
        this, rscope, purity, abi, lifetimes, None, decl);
    b
}

fn ty_of_method_or_bare_fn<AC:AstConv,RS:region_scope + Copy + 'static>(
    this: &AC,
    rscope: &RS,
    purity: ast::purity,
    abi: AbiSet,
    lifetimes: &OptVec<ast::Lifetime>,
    opt_self_info: Option<&SelfInfo>,
    decl: &ast::fn_decl) -> (Option<Option<ty::t>>, ty::BareFnTy)
{
    debug!("ty_of_bare_fn");

    // new region names that appear inside of the fn decl are bound to
    // that function type
    let bound_lifetime_names = bound_lifetimes(this, lifetimes);
    let rb = in_binding_rscope(rscope, RegionParamNames(copy bound_lifetime_names));

    let opt_transformed_self_ty = opt_self_info.map(|&self_info| {
        transform_self_ty(this, &rb, self_info)
    });

    let input_tys = decl.inputs.map(|a| ty_of_arg(this, &rb, *a, None));

    let output_ty = match decl.output.node {
        ast::ty_infer => this.ty_infer(decl.output.span),
        _ => ast_ty_to_ty(this, &rb, decl.output)
    };

    return (opt_transformed_self_ty,
            ty::BareFnTy {
                purity: purity,
                abis: abi,
                sig: ty::FnSig {bound_lifetime_names: bound_lifetime_names,
                                inputs: input_tys,
                                output: output_ty}
            });

    fn transform_self_ty<AC:AstConv,RS:region_scope + Copy + 'static>(
        this: &AC,
        rscope: &RS,
        self_info: &SelfInfo) -> Option<ty::t>
    {
        match self_info.explicit_self.node {
            ast::sty_static => None,
            ast::sty_value => {
                Some(self_info.untransformed_self_ty)
            }
            ast::sty_region(lifetime, mutability) => {
                let region =
                    ast_region_to_region(this, rscope,
                                         self_info.explicit_self.span,
                                         lifetime);
                Some(ty::mk_rptr(this.tcx(), region,
                                 ty::mt {ty: self_info.untransformed_self_ty,
                                         mutbl: mutability}))
            }
            ast::sty_box(mutability) => {
                Some(ty::mk_box(this.tcx(),
                                ty::mt {ty: self_info.untransformed_self_ty,
                                        mutbl: mutability}))
            }
            ast::sty_uniq(mutability) => {
                Some(ty::mk_uniq(this.tcx(),
                                 ty::mt {ty: self_info.untransformed_self_ty,
                                         mutbl: mutability}))
            }
        }
    }
}

pub fn ty_of_closure<AC:AstConv,RS:region_scope + Copy + 'static>(
    this: &AC,
    rscope: &RS,
    sigil: ast::Sigil,
    purity: ast::purity,
    onceness: ast::Onceness,
    bounds: ty::BuiltinBounds,
    opt_lifetime: Option<@ast::Lifetime>,
    decl: &ast::fn_decl,
    expected_sig: Option<ty::FnSig>,
    lifetimes: &OptVec<ast::Lifetime>,
    span: span)
    -> ty::ClosureTy
{
    // The caller should not both provide explicit bound lifetime
    // names and expected types.  Either we infer the bound lifetime
    // names or they are provided, but not both.
    assert!(lifetimes.is_empty() || expected_sig.is_none());

    debug!("ty_of_fn_decl");
    let _i = indenter();

    // resolve the function bound region in the original region
    // scope `rscope`, not the scope of the function parameters
    let bound_region = match opt_lifetime {
        Some(_) => {
            ast_region_to_region(this, rscope, span, opt_lifetime)
        }
        None => {
            match sigil {
                ast::OwnedSigil | ast::ManagedSigil => {
                    // @fn(), ~fn() default to static as the bound
                    // on their upvars:
                    ty::re_static
                }
                ast::BorrowedSigil => {
                    // &fn() defaults as normal for an omitted lifetime:
                    ast_region_to_region(this, rscope, span, opt_lifetime)
                }
            }
        }
    };

    // new region names that appear inside of the fn decl are bound to
    // that function type
    let bound_lifetime_names = bound_lifetimes(this, lifetimes);
    let rb = in_binding_rscope(rscope, RegionParamNames(copy bound_lifetime_names));

    let input_tys = do decl.inputs.mapi |i, a| {
        let expected_arg_ty = do expected_sig.chain_ref |e| {
            // no guarantee that the correct number of expected args
            // were supplied
            if i < e.inputs.len() {Some(e.inputs[i])} else {None}
        };
        ty_of_arg(this, &rb, *a, expected_arg_ty)
    };

    let expected_ret_ty = expected_sig.map(|e| e.output);
    let output_ty = match decl.output.node {
        ast::ty_infer if expected_ret_ty.is_some() => expected_ret_ty.get(),
        ast::ty_infer => this.ty_infer(decl.output.span),
        _ => ast_ty_to_ty(this, &rb, decl.output)
    };

    ty::ClosureTy {
        purity: purity,
        sigil: sigil,
        onceness: onceness,
        region: bound_region,
        bounds: bounds,
        sig: ty::FnSig {bound_lifetime_names: bound_lifetime_names,
                        inputs: input_tys,
                        output: output_ty}
    }
}

fn conv_builtin_bounds(tcx: ty::ctxt, ast_bounds: &Option<OptVec<ast::TyParamBound>>,
                       store: ty::TraitStore)
                       -> ty::BuiltinBounds {
    //! Converts a list of bounds from the AST into a `BuiltinBounds`
    //! struct. Reports an error if any of the bounds that appear
    //! in the AST refer to general traits and not the built-in traits
    //! like `Copy` or `Owned`. Used to translate the bounds that
    //! appear in closure and trait types, where only builtin bounds are
    //! legal.
    //! If no bounds were specified, we choose a "default" bound based on
    //! the allocation type of the fn/trait, as per issue #7264. The user can
    //! override this with an empty bounds list, e.g. "~fn:()" or "~Trait:".

    match (ast_bounds, store) {
        (&Some(ref bound_vec), _) => {
            let mut builtin_bounds = ty::EmptyBuiltinBounds();
            for bound_vec.iter().advance |ast_bound| {
                match *ast_bound {
                    ast::TraitTyParamBound(b) => {
                        match lookup_def_tcx(tcx, b.path.span, b.ref_id) {
                            ast::def_trait(trait_did) => {
                                if try_add_builtin_trait(tcx,
                                                         trait_did,
                                                         &mut builtin_bounds) {
                                    loop; // success
                                }
                            }
                            _ => { }
                        }
                        tcx.sess.span_fatal(
                            b.path.span,
                            fmt!("only the builtin traits can be used \
                                  as closure or object bounds"));
                    }
                    ast::RegionTyParamBound => {
                        builtin_bounds.add(ty::BoundStatic);
                    }
                }
            }
            builtin_bounds
        },
        // ~Trait is sugar for ~Trait:Owned.
        (&None, ty::UniqTraitStore) => {
            let mut set = ty::EmptyBuiltinBounds(); set.add(ty::BoundOwned); set
        }
        // @Trait is sugar for @Trait:'static.
        // &'static Trait is sugar for &'static Trait:'static.
        (&None, ty::BoxTraitStore) |
        (&None, ty::RegionTraitStore(ty::re_static)) => {
            let mut set = ty::EmptyBuiltinBounds(); set.add(ty::BoundStatic); set
        }
        // &'r Trait is sugar for &'r Trait:<no-bounds>.
        (&None, ty::RegionTraitStore(*)) => ty::EmptyBuiltinBounds(),
    }
}

pub fn try_add_builtin_trait(tcx: ty::ctxt,
                             trait_def_id: ast::def_id,
                             builtin_bounds: &mut ty::BuiltinBounds) -> bool {
    //! Checks whether `trait_ref` refers to one of the builtin
    //! traits, like `Copy` or `Owned`, and adds the corresponding
    //! bound to the set `builtin_bounds` if so. Returns true if `trait_ref`
    //! is a builtin trait.

    let li = &tcx.lang_items;
    if trait_def_id == li.owned_trait() {
        builtin_bounds.add(ty::BoundOwned);
        true
    } else if trait_def_id == li.copy_trait() {
        builtin_bounds.add(ty::BoundCopy);
        true
    } else if trait_def_id == li.const_trait() {
        builtin_bounds.add(ty::BoundConst);
        true
    } else if trait_def_id == li.sized_trait() {
        builtin_bounds.add(ty::BoundSized);
        true
    } else {
        false
    }
}
