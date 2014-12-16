// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Code for type-checking closure expressions.

use super::check_fn;
use super::{Expectation, ExpectCastableToType, ExpectHasType, NoExpectation};
use super::FnCtxt;

use astconv;
use middle::infer;
use middle::subst;
use middle::ty::{mod, Ty};
use rscope::RegionScope;
use syntax::abi;
use syntax::ast;
use syntax::ast_util;
use util::ppaux::Repr;

pub fn check_expr_closure<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>,
                                   expr: &ast::Expr,
                                   opt_kind: Option<ast::UnboxedClosureKind>,
                                   decl: &ast::FnDecl,
                                   body: &ast::Block,
                                   expected: Expectation<'tcx>) {
    debug!("check_expr_closure(expr={},expected={})",
           expr.repr(fcx.tcx()),
           expected.repr(fcx.tcx()));

    match opt_kind {
        None => {
            // If users didn't specify what sort of closure they want,
            // examine the expected type. For now, if we see explicit
            // evidence than an unboxed closure is desired, we'll use
            // that, otherwise we'll fall back to boxed closures.
            match deduce_unboxed_closure_expectations_from_expectation(fcx, expected) {
                None => { // doesn't look like an unboxed closure
                    let region = astconv::opt_ast_region_to_region(fcx,
                                                                   fcx.infcx(),
                                                                   expr.span,
                                                                   &None);
                    check_boxed_closure(fcx,
                                        expr,
                                        ty::RegionTraitStore(region, ast::MutMutable),
                                        decl,
                                        body,
                                        expected);
                }
                Some((sig, kind)) => {
                    check_unboxed_closure(fcx, expr, kind, decl, body, Some(sig));
                }
            }
        }

        Some(kind) => {
            let kind = match kind {
                ast::FnUnboxedClosureKind => ty::FnUnboxedClosureKind,
                ast::FnMutUnboxedClosureKind => ty::FnMutUnboxedClosureKind,
                ast::FnOnceUnboxedClosureKind => ty::FnOnceUnboxedClosureKind,
            };

            let expected_sig =
                deduce_unboxed_closure_expectations_from_expectation(fcx, expected)
                .map(|t| t.0);

            check_unboxed_closure(fcx, expr, kind, decl, body, expected_sig);
        }
    }
}

fn check_unboxed_closure<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>,
                                  expr: &ast::Expr,
                                  kind: ty::UnboxedClosureKind,
                                  decl: &ast::FnDecl,
                                  body: &ast::Block,
                                  expected_sig: Option<ty::FnSig<'tcx>>) {
    let expr_def_id = ast_util::local_def(expr.id);

    debug!("check_unboxed_closure kind={} expected_sig={}",
           kind,
           expected_sig.repr(fcx.tcx()));

    let mut fn_ty = astconv::ty_of_closure(
        fcx,
        ast::Unsafety::Normal,
        ast::Many,

        // The `RegionTraitStore` and region_existential_bounds
        // are lies, but we ignore them so it doesn't matter.
        //
        // FIXME(pcwalton): Refactor this API.
        ty::region_existential_bound(ty::ReStatic),
        ty::RegionTraitStore(ty::ReStatic, ast::MutImmutable),

        decl,
        abi::RustCall,
        expected_sig);

    let region = match fcx.infcx().anon_regions(expr.span, 1) {
        Err(_) => {
            fcx.ccx.tcx.sess.span_bug(expr.span,
                                      "can't make anon regions here?!")
        }
        Ok(regions) => regions[0],
    };

    let closure_type = ty::mk_unboxed_closure(fcx.ccx.tcx,
                                              expr_def_id,
                                              region,
                                              fcx.inh.param_env.free_substs.clone());

    fcx.write_ty(expr.id, closure_type);

    check_fn(fcx.ccx,
             ast::Unsafety::Normal,
             expr.id,
             &fn_ty.sig,
             decl,
             expr.id,
             &*body,
             fcx.inh);

    // Tuple up the arguments and insert the resulting function type into
    // the `unboxed_closures` table.
    fn_ty.sig.0.inputs = vec![ty::mk_tup(fcx.tcx(), fn_ty.sig.0.inputs)];

    debug!("unboxed_closure for {} --> sig={} kind={}",
           expr_def_id.repr(fcx.tcx()),
           fn_ty.sig.repr(fcx.tcx()),
           kind);

    let unboxed_closure = ty::UnboxedClosure {
        closure_type: fn_ty,
        kind: kind,
    };

    fcx.inh
        .unboxed_closures
        .borrow_mut()
        .insert(expr_def_id, unboxed_closure);
}

fn deduce_unboxed_closure_expectations_from_expectation<'a,'tcx>(
    fcx: &FnCtxt<'a,'tcx>,
    expected: Expectation<'tcx>)
    -> Option<(ty::FnSig<'tcx>,ty::UnboxedClosureKind)>
{
    match expected.resolve(fcx) {
        NoExpectation => None,
        ExpectCastableToType(t) | ExpectHasType(t) => {
            deduce_unboxed_closure_expectations_from_expected_type(fcx, t)
        }
    }
}

fn deduce_unboxed_closure_expectations_from_expected_type<'a,'tcx>(
    fcx: &FnCtxt<'a,'tcx>,
    expected_ty: Ty<'tcx>)
    -> Option<(ty::FnSig<'tcx>,ty::UnboxedClosureKind)>
{
    match expected_ty.sty {
        ty::ty_trait(ref object_type) => {
            deduce_unboxed_closure_expectations_from_trait_ref(fcx, &object_type.principal)
        }
        ty::ty_infer(ty::TyVar(vid)) => {
            deduce_unboxed_closure_expectations_from_obligations(fcx, vid)
        }
        _ => {
            None
        }
    }
}

fn deduce_unboxed_closure_expectations_from_trait_ref<'a,'tcx>(
    fcx: &FnCtxt<'a,'tcx>,
    trait_ref: &ty::PolyTraitRef<'tcx>)
    -> Option<(ty::FnSig<'tcx>, ty::UnboxedClosureKind)>
{
    let tcx = fcx.tcx();

    debug!("deduce_unboxed_closure_expectations_from_object_type({})",
           trait_ref.repr(tcx));

    let kind = match tcx.lang_items.fn_trait_kind(trait_ref.def_id()) {
        Some(k) => k,
        None => { return None; }
    };

    debug!("found object type {}", kind);

    let arg_param_ty = *trait_ref.substs().types.get(subst::TypeSpace, 0);
    let arg_param_ty = fcx.infcx().resolve_type_vars_if_possible(&arg_param_ty);
    debug!("arg_param_ty {}", arg_param_ty.repr(tcx));

    let input_tys = match arg_param_ty.sty {
        ty::ty_tup(ref tys) => { (*tys).clone() }
        _ => { return None; }
    };
    debug!("input_tys {}", input_tys.repr(tcx));

    let ret_param_ty = *trait_ref.substs().types.get(subst::TypeSpace, 1);
    let ret_param_ty = fcx.infcx().resolve_type_vars_if_possible(&ret_param_ty);
    debug!("ret_param_ty {}", ret_param_ty.repr(tcx));

    let fn_sig = ty::FnSig {
        inputs: input_tys,
        output: ty::FnConverging(ret_param_ty),
        variadic: false
    };
    debug!("fn_sig {}", fn_sig.repr(tcx));

    return Some((fn_sig, kind));
}

fn deduce_unboxed_closure_expectations_from_obligations<'a,'tcx>(
    fcx: &FnCtxt<'a,'tcx>,
    expected_vid: ty::TyVid)
    -> Option<(ty::FnSig<'tcx>, ty::UnboxedClosureKind)>
{
    // Here `expected_ty` is known to be a type inference variable.
    for obligation in fcx.inh.fulfillment_cx.borrow().pending_obligations().iter() {
        match obligation.trait_ref {
            ty::Predicate::Trait(ref trait_ref) => {
                let self_ty = fcx.infcx().shallow_resolve(trait_ref.self_ty());
                match self_ty.sty {
                    ty::ty_infer(ty::TyVar(v)) if expected_vid == v => { }
                    _ => { continue; }
                }

                match deduce_unboxed_closure_expectations_from_trait_ref(fcx, &**trait_ref) {
                    Some(e) => { return Some(e); }
                    None => { }
                }
            }
            ty::Predicate::Equate(..) |
            ty::Predicate::RegionOutlives(..) |
            ty::Predicate::TypeOutlives(..) => {
            }
        }
    }

    None
}


fn check_boxed_closure<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>,
                                expr: &ast::Expr,
                                store: ty::TraitStore,
                                decl: &ast::FnDecl,
                                body: &ast::Block,
                                expected: Expectation<'tcx>) {
    let tcx = fcx.ccx.tcx;

    // Find the expected input/output types (if any). Substitute
    // fresh bound regions for any bound regions we find in the
    // expected types so as to avoid capture.
    let expected_cenv = expected.map_to_option(fcx, |ty| match ty.sty {
        ty::ty_closure(ref cenv) => Some(cenv),
        _ => None
    });
    let (expected_sig, expected_onceness, expected_bounds) = match expected_cenv {
        Some(cenv) => {
            let (sig, _) =
                ty::replace_late_bound_regions(
                    tcx,
                    &cenv.sig,
                    |_, debruijn| fcx.inh.infcx.fresh_bound_region(debruijn));
            let onceness = match (&store, &cenv.store) {
                // As the closure type and onceness go, only three
                // combinations are legit:
                //      once closure
                //      many closure
                //      once proc
                // If the actual and expected closure type disagree with
                // each other, set expected onceness to be always Once or
                // Many according to the actual type. Otherwise, it will
                // yield either an illegal "many proc" or a less known
                // "once closure" in the error message.
                (&ty::UniqTraitStore, &ty::UniqTraitStore) |
                (&ty::RegionTraitStore(..), &ty::RegionTraitStore(..)) =>
                    cenv.onceness,
                (&ty::UniqTraitStore, _) => ast::Once,
                (&ty::RegionTraitStore(..), _) => ast::Many,
            };
            (Some(sig), onceness, cenv.bounds)
        }
        _ => {
            // Not an error! Means we're inferring the closure type
            let region = fcx.infcx().next_region_var(
                infer::AddrOfRegion(expr.span));
            let bounds = ty::region_existential_bound(region);
            let onceness = ast::Many;
            (None, onceness, bounds)
        }
    };

    // construct the function type
    let fn_ty = astconv::ty_of_closure(fcx,
                                       ast::Unsafety::Normal,
                                       expected_onceness,
                                       expected_bounds,
                                       store,
                                       decl,
                                       abi::Rust,
                                       expected_sig);
    let fty_sig = fn_ty.sig.clone();
    let fty = ty::mk_closure(tcx, fn_ty);
    debug!("check_expr_fn fty={}", fcx.infcx().ty_to_string(fty));

    fcx.write_ty(expr.id, fty);

    // If the closure is a stack closure and hasn't had some non-standard
    // style inferred for it, then check it under its parent's style.
    // Otherwise, use its own
    let (inherited_style, inherited_style_id) = match store {
        ty::RegionTraitStore(..) => (fcx.ps.borrow().unsafety,
                                     fcx.ps.borrow().def),
        ty::UniqTraitStore => (ast::Unsafety::Normal, expr.id)
    };

    check_fn(fcx.ccx,
             inherited_style,
             inherited_style_id,
             &fty_sig,
             &*decl,
             expr.id,
             &*body,
             fcx.inh);
}
