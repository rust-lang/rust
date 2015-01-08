// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::autoderef;
use super::AutorefArgs;
use super::check_argument_types;
use super::check_expr;
use super::check_method_argument_types;
use super::err_args;
use super::FnCtxt;
use super::LvaluePreference;
use super::method;
use super::structurally_resolved_type;
use super::TupleArgumentsFlag;
use super::write_call;

use middle::infer;
use middle::ty::{self, Ty};
use syntax::ast;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::ptr::P;
use CrateCtxt;

/// Check that it is legal to call methods of the trait corresponding
/// to `trait_id` (this only cares about the trait, not the specific
/// method that is called)
pub fn check_legal_trait_for_method_call(ccx: &CrateCtxt, span: Span, trait_id: ast::DefId) {
    let tcx = ccx.tcx;
    let did = Some(trait_id);
    let li = &tcx.lang_items;

    if did == li.drop_trait() {
        span_err!(tcx.sess, span, E0040, "explicit use of destructor method");
    } else if !tcx.sess.features.borrow().unboxed_closures {
        // the #[feature(unboxed_closures)] feature isn't
        // activated so we need to enforce the closure
        // restrictions.

        let method = if did == li.fn_trait() {
            "call"
        } else if did == li.fn_mut_trait() {
            "call_mut"
        } else if did == li.fn_once_trait() {
            "call_once"
        } else {
            return // not a closure method, everything is OK.
        };

        span_err!(tcx.sess, span, E0174,
                  "explicit use of unboxed closure method `{}` is experimental",
                  method);
        span_help!(tcx.sess, span,
                   "add `#![feature(unboxed_closures)]` to the crate attributes to enable");
    }
}

pub fn check_call<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                            call_expr: &ast::Expr,
                            callee_expr: &ast::Expr,
                            arg_exprs: &[P<ast::Expr>])
{
    check_expr(fcx, callee_expr);
    let original_callee_ty = fcx.expr_ty(callee_expr);
    let (callee_ty, _, result) =
        autoderef(fcx,
                  callee_expr.span,
                  original_callee_ty,
                  Some(callee_expr),
                  LvaluePreference::NoPreference,
                  |adj_ty, idx| {
                      let autoderefref = ty::AutoDerefRef { autoderefs: idx, autoref: None };
                      try_overloaded_call_step(fcx, call_expr, callee_expr,
                                               adj_ty, autoderefref)
                  });

    match result {
        None => {
            // this will report an error since original_callee_ty is not a fn
            confirm_builtin_call(fcx, call_expr, original_callee_ty, arg_exprs);
        }

        Some(CallStep::Builtin) => {
            confirm_builtin_call(fcx, call_expr, callee_ty, arg_exprs);
        }

        Some(CallStep::Overloaded(method_callee)) => {
            confirm_overloaded_call(fcx, call_expr, arg_exprs, method_callee);
        }
    }
}

enum CallStep<'tcx> {
    Builtin,
    Overloaded(ty::MethodCallee<'tcx>)
}

fn try_overloaded_call_step<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                      call_expr: &ast::Expr,
                                      callee_expr: &ast::Expr,
                                      adjusted_ty: Ty<'tcx>,
                                      autoderefref: ty::AutoDerefRef<'tcx>)
                                      -> Option<CallStep<'tcx>>
{
    // If the callee is a bare function or a closure, then we're all set.
    match structurally_resolved_type(fcx, callee_expr.span, adjusted_ty).sty {
        ty::ty_bare_fn(..) => {
            fcx.write_adjustment(callee_expr.id,
                                 callee_expr.span,
                                 ty::AdjustDerefRef(autoderefref));
            return Some(CallStep::Builtin);
        }

        _ => {}
    }

    // Try the options that are least restrictive on the caller first.
    for &(opt_trait_def_id, method_name) in [
        (fcx.tcx().lang_items.fn_trait(), token::intern("call")),
        (fcx.tcx().lang_items.fn_mut_trait(), token::intern("call_mut")),
        (fcx.tcx().lang_items.fn_once_trait(), token::intern("call_once")),
    ].iter() {
        let trait_def_id = match opt_trait_def_id {
            Some(def_id) => def_id,
            None => continue,
        };

        match method::lookup_in_trait_adjusted(fcx,
                                               call_expr.span,
                                               Some(&*callee_expr),
                                               method_name,
                                               trait_def_id,
                                               autoderefref.clone(),
                                               adjusted_ty,
                                               None) {
            None => continue,
            Some(method_callee) => {
                return Some(CallStep::Overloaded(method_callee));
            }
        }
    }

    None
}

fn confirm_builtin_call<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>,
                                 call_expr: &ast::Expr,
                                 callee_ty: Ty<'tcx>,
                                 arg_exprs: &[P<ast::Expr>])
{
    let error_fn_sig;

    let fn_sig = match callee_ty.sty {
        ty::ty_bare_fn(_, &ty::BareFnTy {ref sig, ..}) => {
            sig
        }
        _ => {
            fcx.type_error_message(call_expr.span, |actual| {
                format!("expected function, found `{}`", actual)
            }, callee_ty, None);

            // This is the "default" function signature, used in case of error.
            // In that case, we check each argument against "error" in order to
            // set up all the node type bindings.
            error_fn_sig = ty::Binder(ty::FnSig {
                inputs: err_args(fcx.tcx(), arg_exprs.len()),
                output: ty::FnConverging(fcx.tcx().types.err),
                variadic: false
            });

            &error_fn_sig
        }
    };

    // Replace any late-bound regions that appear in the function
    // signature with region variables. We also have to
    // renormalize the associated types at this point, since they
    // previously appeared within a `Binder<>` and hence would not
    // have been normalized before.
    let fn_sig =
        fcx.infcx().replace_late_bound_regions_with_fresh_var(call_expr.span,
                                                              infer::FnCall,
                                                              fn_sig).0;
    let fn_sig =
        fcx.normalize_associated_types_in(call_expr.span, &fn_sig);

    // Call the generic checker.
    let arg_exprs: Vec<_> = arg_exprs.iter().collect(); // for some weird reason we take &[&P<...>].
    check_argument_types(fcx,
                         call_expr.span,
                         fn_sig.inputs.as_slice(),
                         arg_exprs.as_slice(),
                         AutorefArgs::No,
                         fn_sig.variadic,
                         TupleArgumentsFlag::DontTupleArguments);

    write_call(fcx, call_expr, fn_sig.output);
}

fn confirm_overloaded_call<'a,'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                    call_expr: &ast::Expr,
                                    arg_exprs: &[P<ast::Expr>],
                                    method_callee: ty::MethodCallee<'tcx>)
{
    let arg_exprs: Vec<_> = arg_exprs.iter().collect(); // for some weird reason we take &[&P<...>].
    let output_type = check_method_argument_types(fcx,
                                                  call_expr.span,
                                                  method_callee.ty,
                                                  call_expr,
                                                  arg_exprs.as_slice(),
                                                  AutorefArgs::No,
                                                  TupleArgumentsFlag::TupleArguments);
    let method_call = ty::MethodCall::expr(call_expr.id);
    fcx.inh.method_map.borrow_mut().insert(method_call, method_callee);
    write_call(fcx, call_expr, output_type);
}

