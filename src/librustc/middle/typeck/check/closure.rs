// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Code for type-checking closure expressions.
 */

use super::check_fn;
use super::Expectation;
use super::FnCtxt;

use middle::ty;
use middle::typeck::astconv;
use middle::typeck::infer;
use middle::typeck::rscope::RegionScope;
use syntax::abi;
use syntax::ast;
use syntax::ast_util;
use util::ppaux::Repr;

pub fn check_unboxed_closure(fcx: &FnCtxt,
                             expr: &ast::Expr,
                             kind: ast::UnboxedClosureKind,
                             decl: &ast::FnDecl,
                             body: &ast::Block) {
    let expr_def_id = ast_util::local_def(expr.id);

    let mut fn_ty = astconv::ty_of_closure(
        fcx,
        ast::NormalFn,
        ast::Many,

        // The `RegionTraitStore` and region_existential_bounds
        // are lies, but we ignore them so it doesn't matter.
        //
        // FIXME(pcwalton): Refactor this API.
        ty::region_existential_bound(ty::ReStatic),
        ty::RegionTraitStore(ty::ReStatic, ast::MutImmutable),

        decl,
        abi::RustCall,
        None);

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
             ast::NormalFn,
             expr.id,
             &fn_ty.sig,
             decl,
             expr.id,
             &*body,
             fcx.inh);

    // Tuple up the arguments and insert the resulting function type into
    // the `unboxed_closures` table.
    fn_ty.sig.inputs = vec![ty::mk_tup(fcx.tcx(), fn_ty.sig.inputs)];

    let kind = match kind {
        ast::FnUnboxedClosureKind => ty::FnUnboxedClosureKind,
        ast::FnMutUnboxedClosureKind => ty::FnMutUnboxedClosureKind,
        ast::FnOnceUnboxedClosureKind => ty::FnOnceUnboxedClosureKind,
    };

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

pub fn check_expr_fn<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>,
                              expr: &ast::Expr,
                              store: ty::TraitStore,
                              decl: &ast::FnDecl,
                              body: &ast::Block,
                              expected: Expectation<'tcx>) {
    let tcx = fcx.ccx.tcx;

    // Find the expected input/output types (if any). Substitute
    // fresh bound regions for any bound regions we find in the
    // expected types so as to avoid capture.
    let expected_sty = expected.map_to_option(fcx, |x| Some((*x).clone()));
    let (expected_sig,
         expected_onceness,
         expected_bounds) = {
        match expected_sty {
            Some(ty::ty_closure(ref cenv)) => {
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
                let (bounds, onceness) = match expr.node {
                    ast::ExprProc(..) => {
                        let mut bounds = ty::region_existential_bound(ty::ReStatic);
                        bounds.builtin_bounds.insert(ty::BoundSend); // FIXME
                        (bounds, ast::Once)
                    }
                    _ => {
                        let region = fcx.infcx().next_region_var(
                            infer::AddrOfRegion(expr.span));
                        (ty::region_existential_bound(region), ast::Many)
                    }
                };
                (None, onceness, bounds)
            }
        }
    };

    // construct the function type
    let fn_ty = astconv::ty_of_closure(fcx,
                                       ast::NormalFn,
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
        ty::RegionTraitStore(..) => (fcx.ps.borrow().fn_style,
                                     fcx.ps.borrow().def),
        ty::UniqTraitStore => (ast::NormalFn, expr.id)
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
