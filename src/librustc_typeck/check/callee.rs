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
use super::check_argument_types;
use super::check_expr;
use super::check_method_argument_types;
use super::demand;
use super::DeferredCallResolution;
use super::err_args;
use super::Expectation;
use super::expected_types_for_fn_args;
use super::FnCtxt;
use super::method;
use super::structurally_resolved_type;
use super::TupleArgumentsFlag;
use super::UnresolvedTypeAction;
use super::write_call;

use CrateCtxt;
use middle::cstore::LOCAL_CRATE;
use middle::def;
use middle::def_id::DefId;
use middle::infer;
use middle::ty::{self, LvaluePreference, Ty};
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::ptr::P;

use rustc_front::hir;

/// Check that it is legal to call methods of the trait corresponding
/// to `trait_id` (this only cares about the trait, not the specific
/// method that is called)
pub fn check_legal_trait_for_method_call(ccx: &CrateCtxt, span: Span, trait_id: DefId) {
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
        fileline_help!(tcx.sess, span,
                   "add `#![feature(unboxed_closures)]` to the crate attributes to enable");
    }
}

pub fn check_call<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                            call_expr: &'tcx hir::Expr,
                            callee_expr: &'tcx hir::Expr,
                            arg_exprs: &'tcx [P<hir::Expr>],
                            expected: Expectation<'tcx>)
{
    check_expr(fcx, callee_expr);
    let original_callee_ty = fcx.expr_ty(callee_expr);
    let (callee_ty, _, result) =
        autoderef(fcx,
                  callee_expr.span,
                  original_callee_ty,
                  Some(callee_expr),
                  UnresolvedTypeAction::Error,
                  LvaluePreference::NoPreference,
                  |adj_ty, idx| {
                      try_overloaded_call_step(fcx, call_expr, callee_expr, adj_ty, idx)
                  });

    match result {
        None => {
            // this will report an error since original_callee_ty is not a fn
            confirm_builtin_call(fcx, call_expr, original_callee_ty, arg_exprs, expected);
        }

        Some(CallStep::Builtin) => {
            confirm_builtin_call(fcx, call_expr, callee_ty, arg_exprs, expected);
        }

        Some(CallStep::DeferredClosure(fn_sig)) => {
            confirm_deferred_closure_call(fcx, call_expr, arg_exprs, expected, fn_sig);
        }

        Some(CallStep::Overloaded(method_callee)) => {
            confirm_overloaded_call(fcx, call_expr, callee_expr,
                                    arg_exprs, expected, method_callee);
        }
    }
}

enum CallStep<'tcx> {
    Builtin,
    DeferredClosure(ty::FnSig<'tcx>),
    Overloaded(ty::MethodCallee<'tcx>)
}

fn try_overloaded_call_step<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                      call_expr: &'tcx hir::Expr,
                                      callee_expr: &'tcx hir::Expr,
                                      adjusted_ty: Ty<'tcx>,
                                      autoderefs: usize)
                                      -> Option<CallStep<'tcx>>
{
    debug!("try_overloaded_call_step(call_expr={:?}, adjusted_ty={:?}, autoderefs={})",
           call_expr,
           adjusted_ty,
           autoderefs);

    // If the callee is a bare function or a closure, then we're all set.
    match structurally_resolved_type(fcx, callee_expr.span, adjusted_ty).sty {
        ty::TyBareFn(..) => {
            fcx.write_autoderef_adjustment(callee_expr.id, autoderefs);
            return Some(CallStep::Builtin);
        }

        ty::TyClosure(def_id, ref substs) => {
            assert_eq!(def_id.krate, LOCAL_CRATE);

            // Check whether this is a call to a closure where we
            // haven't yet decided on whether the closure is fn vs
            // fnmut vs fnonce. If so, we have to defer further processing.
            if fcx.infcx().closure_kind(def_id).is_none() {
                let closure_ty =
                    fcx.infcx().closure_type(def_id, substs);
                let fn_sig =
                    fcx.infcx().replace_late_bound_regions_with_fresh_var(call_expr.span,
                                                                          infer::FnCall,
                                                                          &closure_ty.sig).0;
                fcx.record_deferred_call_resolution(def_id, Box::new(CallResolution {
                    call_expr: call_expr,
                    callee_expr: callee_expr,
                    adjusted_ty: adjusted_ty,
                    autoderefs: autoderefs,
                    fn_sig: fn_sig.clone(),
                    closure_def_id: def_id
                }));
                return Some(CallStep::DeferredClosure(fn_sig));
            }
        }

        // Hack: we know that there are traits implementing Fn for &F
        // where F:Fn and so forth. In the particular case of types
        // like `x: &mut FnMut()`, if there is a call `x()`, we would
        // normally translate to `FnMut::call_mut(&mut x, ())`, but
        // that winds up requiring `mut x: &mut FnMut()`. A little
        // over the top. The simplest fix by far is to just ignore
        // this case and deref again, so we wind up with
        // `FnMut::call_mut(&mut *x, ())`.
        ty::TyRef(..) if autoderefs == 0 => {
            return None;
        }

        _ => {}
    }

    try_overloaded_call_traits(fcx, call_expr, callee_expr, adjusted_ty, autoderefs)
        .map(|method_callee| CallStep::Overloaded(method_callee))
}

fn try_overloaded_call_traits<'a,'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                       call_expr: &hir::Expr,
                                       callee_expr: &hir::Expr,
                                       adjusted_ty: Ty<'tcx>,
                                       autoderefs: usize)
                                       -> Option<ty::MethodCallee<'tcx>>
{
    // Try the options that are least restrictive on the caller first.
    for &(opt_trait_def_id, method_name) in &[
        (fcx.tcx().lang_items.fn_trait(), token::intern("call")),
        (fcx.tcx().lang_items.fn_mut_trait(), token::intern("call_mut")),
        (fcx.tcx().lang_items.fn_once_trait(), token::intern("call_once")),
    ] {
        let trait_def_id = match opt_trait_def_id {
            Some(def_id) => def_id,
            None => continue,
        };

        match method::lookup_in_trait_adjusted(fcx,
                                               call_expr.span,
                                               Some(&*callee_expr),
                                               method_name,
                                               trait_def_id,
                                               autoderefs,
                                               false,
                                               adjusted_ty,
                                               None) {
            None => continue,
            Some(method_callee) => {
                return Some(method_callee);
            }
        }
    }

    None
}

fn confirm_builtin_call<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>,
                                 call_expr: &hir::Expr,
                                 callee_ty: Ty<'tcx>,
                                 arg_exprs: &'tcx [P<hir::Expr>],
                                 expected: Expectation<'tcx>)
{
    let error_fn_sig;

    let fn_sig = match callee_ty.sty {
        ty::TyBareFn(_, &ty::BareFnTy {ref sig, ..}) => {
            sig
        }
        _ => {
            fcx.type_error_message(call_expr.span, |actual| {
                format!("expected function, found `{}`", actual)
            }, callee_ty, None);

            if let hir::ExprCall(ref expr, _) = call_expr.node {
                let tcx = fcx.tcx();
                if let Some(pr) = tcx.def_map.borrow().get(&expr.id) {
                    if pr.depth == 0 && pr.base_def != def::DefErr {
                        if let Some(span) = tcx.map.span_if_local(pr.def_id()) {
                            tcx.sess.span_note(span, "defined here")
                        }
                    }
                }
            }

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
    let expected_arg_tys = expected_types_for_fn_args(fcx,
                                                      call_expr.span,
                                                      expected,
                                                      fn_sig.output,
                                                      &fn_sig.inputs);
    check_argument_types(fcx,
                         call_expr.span,
                         &fn_sig.inputs,
                         &expected_arg_tys[..],
                         arg_exprs,
                         fn_sig.variadic,
                         TupleArgumentsFlag::DontTupleArguments);

    write_call(fcx, call_expr, fn_sig.output);
}

fn confirm_deferred_closure_call<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>,
                                          call_expr: &hir::Expr,
                                          arg_exprs: &'tcx [P<hir::Expr>],
                                          expected: Expectation<'tcx>,
                                          fn_sig: ty::FnSig<'tcx>)
{
    // `fn_sig` is the *signature* of the cosure being called. We
    // don't know the full details yet (`Fn` vs `FnMut` etc), but we
    // do know the types expected for each argument and the return
    // type.

    let expected_arg_tys =
        expected_types_for_fn_args(fcx,
                                   call_expr.span,
                                   expected,
                                   fn_sig.output.clone(),
                                   &*fn_sig.inputs);

    check_argument_types(fcx,
                         call_expr.span,
                         &*fn_sig.inputs,
                         &*expected_arg_tys,
                         arg_exprs,
                         fn_sig.variadic,
                         TupleArgumentsFlag::TupleArguments);

    write_call(fcx, call_expr, fn_sig.output);
}

fn confirm_overloaded_call<'a,'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                    call_expr: &hir::Expr,
                                    callee_expr: &'tcx hir::Expr,
                                    arg_exprs: &'tcx [P<hir::Expr>],
                                    expected: Expectation<'tcx>,
                                    method_callee: ty::MethodCallee<'tcx>)
{
    let output_type =
        check_method_argument_types(fcx,
                                    call_expr.span,
                                    method_callee.ty,
                                    callee_expr,
                                    arg_exprs,
                                    TupleArgumentsFlag::TupleArguments,
                                    expected);
    write_call(fcx, call_expr, output_type);

    write_overloaded_call_method_map(fcx, call_expr, method_callee);
}

fn write_overloaded_call_method_map<'a,'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                             call_expr: &hir::Expr,
                                             method_callee: ty::MethodCallee<'tcx>) {
    let method_call = ty::MethodCall::expr(call_expr.id);
    fcx.inh.tables.borrow_mut().method_map.insert(method_call, method_callee);
}

#[derive(Debug)]
struct CallResolution<'tcx> {
    call_expr: &'tcx hir::Expr,
    callee_expr: &'tcx hir::Expr,
    adjusted_ty: Ty<'tcx>,
    autoderefs: usize,
    fn_sig: ty::FnSig<'tcx>,
    closure_def_id: DefId,
}

impl<'tcx> DeferredCallResolution<'tcx> for CallResolution<'tcx> {
    fn resolve<'a>(&mut self, fcx: &FnCtxt<'a,'tcx>) {
        debug!("DeferredCallResolution::resolve() {:?}",
               self);

        // we should not be invoked until the closure kind has been
        // determined by upvar inference
        assert!(fcx.infcx().closure_kind(self.closure_def_id).is_some());

        // We may now know enough to figure out fn vs fnmut etc.
        match try_overloaded_call_traits(fcx, self.call_expr, self.callee_expr,
                                         self.adjusted_ty, self.autoderefs) {
            Some(method_callee) => {
                // One problem is that when we get here, we are going
                // to have a newly instantiated function signature
                // from the call trait. This has to be reconciled with
                // the older function signature we had before. In
                // principle we *should* be able to fn_sigs(), but we
                // can't because of the annoying need for a TypeTrace.
                // (This always bites me, should find a way to
                // refactor it.)
                let method_sig = fcx.tcx().no_late_bound_regions(method_callee.ty.fn_sig())
                                          .unwrap();

                debug!("attempt_resolution: method_callee={:?}",
                       method_callee);

                for (&method_arg_ty, &self_arg_ty) in
                    method_sig.inputs[1..].iter().zip(&self.fn_sig.inputs)
                {
                    demand::eqtype(fcx, self.call_expr.span, self_arg_ty, method_arg_ty);
                }

                let nilty = fcx.tcx().mk_nil();
                demand::eqtype(fcx,
                               self.call_expr.span,
                               method_sig.output.unwrap_or(nilty),
                               self.fn_sig.output.unwrap_or(nilty));

                write_overloaded_call_method_map(fcx, self.call_expr, method_callee);
            }
            None => {
                fcx.tcx().sess.span_bug(
                    self.call_expr.span,
                    "failed to find an overloaded call trait for closure call");
            }
        }
    }
}
