// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{Expectation, FnCtxt, TupleArgumentsFlag};
use super::autoderef::Autoderef;
use super::method::MethodCallee;

use hir::def::Def;
use hir::def_id::{DefId, LOCAL_CRATE};
use rustc::{infer, traits};
use rustc::ty::{self, TyCtxt, TypeFoldable, LvaluePreference, Ty};
use rustc::ty::adjustment::{Adjustment, Adjust, AutoBorrow};
use syntax::abi;
use syntax::symbol::Symbol;
use syntax_pos::Span;

use rustc::hir;

/// Check that it is legal to call methods of the trait corresponding
/// to `trait_id` (this only cares about the trait, not the specific
/// method that is called)
pub fn check_legal_trait_for_method_call(tcx: TyCtxt, span: Span, trait_id: DefId) {
    if tcx.lang_items().drop_trait() == Some(trait_id) {
        struct_span_err!(tcx.sess, span, E0040, "explicit use of destructor method")
            .span_label(span, "explicit destructor calls not allowed")
            .emit();
    }
}

enum CallStep<'tcx> {
    Builtin(Ty<'tcx>),
    DeferredClosure(ty::FnSig<'tcx>),
    /// e.g. enum variant constructors
    Overloaded(MethodCallee<'tcx>),
}

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    pub fn check_call(&self,
                      call_expr: &'gcx hir::Expr,
                      callee_expr: &'gcx hir::Expr,
                      arg_exprs: &'gcx [hir::Expr],
                      expected: Expectation<'tcx>)
                      -> Ty<'tcx> {
        let original_callee_ty = self.check_expr(callee_expr);
        let expr_ty = self.structurally_resolved_type(call_expr.span, original_callee_ty);

        let mut autoderef = self.autoderef(callee_expr.span, expr_ty);
        let mut result = None;
        while result.is_none() && autoderef.next().is_some() {
            result = self.try_overloaded_call_step(call_expr, callee_expr, &autoderef);
        }
        autoderef.finalize();

        let output = match result {
            None => {
                // this will report an error since original_callee_ty is not a fn
                self.confirm_builtin_call(call_expr, original_callee_ty, arg_exprs, expected)
            }

            Some(CallStep::Builtin(callee_ty)) => {
                self.confirm_builtin_call(call_expr, callee_ty, arg_exprs, expected)
            }

            Some(CallStep::DeferredClosure(fn_sig)) => {
                self.confirm_deferred_closure_call(call_expr, arg_exprs, expected, fn_sig)
            }

            Some(CallStep::Overloaded(method_callee)) => {
                self.confirm_overloaded_call(call_expr, arg_exprs, expected, method_callee)
            }
        };

        // we must check that return type of called functions is WF:
        self.register_wf_obligation(output, call_expr.span, traits::MiscObligation);

        output
    }

    fn try_overloaded_call_step(&self,
                                call_expr: &'gcx hir::Expr,
                                callee_expr: &'gcx hir::Expr,
                                autoderef: &Autoderef<'a, 'gcx, 'tcx>)
                                -> Option<CallStep<'tcx>> {
        let adjusted_ty = autoderef.unambiguous_final_ty();
        debug!("try_overloaded_call_step(call_expr={:?}, adjusted_ty={:?})",
               call_expr,
               adjusted_ty);

        // If the callee is a bare function or a closure, then we're all set.
        match adjusted_ty.sty {
            ty::TyFnDef(..) | ty::TyFnPtr(_) => {
                let adjustments = autoderef.adjust_steps(LvaluePreference::NoPreference);
                self.apply_adjustments(callee_expr, adjustments);
                return Some(CallStep::Builtin(adjusted_ty));
            }

            ty::TyClosure(def_id, substs) => {
                assert_eq!(def_id.krate, LOCAL_CRATE);

                // Check whether this is a call to a closure where we
                // haven't yet decided on whether the closure is fn vs
                // fnmut vs fnonce. If so, we have to defer further processing.
                if self.closure_kind(def_id, substs).is_none() {
                    let closure_ty = self.closure_sig(def_id, substs);
                    let fn_sig = self.replace_late_bound_regions_with_fresh_var(call_expr.span,
                                                                   infer::FnCall,
                                                                   &closure_ty)
                        .0;
                    let adjustments = autoderef.adjust_steps(LvaluePreference::NoPreference);
                    self.record_deferred_call_resolution(def_id, DeferredCallResolution {
                        call_expr,
                        callee_expr,
                        adjusted_ty,
                        adjustments,
                        fn_sig,
                        closure_def_id: def_id,
                        closure_substs: substs,
                    });
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
            ty::TyRef(..) if autoderef.step_count() == 0 => {
                return None;
            }

            _ => {}
        }

        self.try_overloaded_call_traits(call_expr, adjusted_ty).map(|(autoref, method)| {
            let mut adjustments = autoderef.adjust_steps(LvaluePreference::NoPreference);
            adjustments.extend(autoref);
            self.apply_adjustments(callee_expr, adjustments);
            CallStep::Overloaded(method)
        })
    }

    fn try_overloaded_call_traits(&self,
                                  call_expr: &hir::Expr,
                                  adjusted_ty: Ty<'tcx>)
                                  -> Option<(Option<Adjustment<'tcx>>,
                                             MethodCallee<'tcx>)> {
        // Try the options that are least restrictive on the caller first.
        for &(opt_trait_def_id, method_name, borrow) in
            &[(self.tcx.lang_items().fn_trait(), Symbol::intern("call"), true),
              (self.tcx.lang_items().fn_mut_trait(), Symbol::intern("call_mut"), true),
              (self.tcx.lang_items().fn_once_trait(), Symbol::intern("call_once"), false)] {
            let trait_def_id = match opt_trait_def_id {
                Some(def_id) => def_id,
                None => continue,
            };

            match self.lookup_method_in_trait(call_expr.span,
                                              method_name,
                                              trait_def_id,
                                              adjusted_ty,
                                              None) {
                None => continue,
                Some(ok) => {
                    let method = self.register_infer_ok_obligations(ok);
                    let mut autoref = None;
                    if borrow {
                        if let ty::TyRef(region, mt) = method.sig.inputs()[0].sty {
                            autoref = Some(Adjustment {
                                kind: Adjust::Borrow(AutoBorrow::Ref(region, mt.mutbl)),
                                target: method.sig.inputs()[0]
                            });
                        }
                    }
                    return Some((autoref, method));
                }
            }
        }

        None
    }

    fn confirm_builtin_call(&self,
                            call_expr: &hir::Expr,
                            callee_ty: Ty<'tcx>,
                            arg_exprs: &'gcx [hir::Expr],
                            expected: Expectation<'tcx>)
                            -> Ty<'tcx> {
        let (fn_sig, def_span) = match callee_ty.sty {
            ty::TyFnDef(def_id, _) => {
                (callee_ty.fn_sig(self.tcx), self.tcx.hir.span_if_local(def_id))
            }
            ty::TyFnPtr(sig) => (sig, None),
            ref t => {
                let mut unit_variant = None;
                if let &ty::TyAdt(adt_def, ..) = t {
                    if adt_def.is_enum() {
                        if let hir::ExprCall(ref expr, _) = call_expr.node {
                            unit_variant = Some(self.tcx.hir.node_to_pretty_string(expr.id))
                        }
                    }
                }
                let mut err = type_error_struct!(self.tcx.sess, call_expr.span, callee_ty, E0618,
                                                 "expected function, found `{}`",
                                                 if let Some(ref path) = unit_variant {
                                                     path.to_string()
                                                 } else {
                                                     callee_ty.to_string()
                                                 });
                if let Some(path) = unit_variant {
                    err.help(&format!("did you mean to write `{}`?", path));
                }

                if let hir::ExprCall(ref expr, _) = call_expr.node {
                    let def = if let hir::ExprPath(ref qpath) = expr.node {
                        self.tables.borrow().qpath_def(qpath, expr.hir_id)
                    } else {
                        Def::Err
                    };
                    let def_span = match def {
                        Def::Err => None,
                        Def::Local(id) | Def::Upvar(id, ..) => {
                            Some(self.tcx.hir.span(id))
                        }
                        _ => self.tcx.hir.span_if_local(def.def_id())
                    };
                    if let Some(span) = def_span {
                        err.span_note(span, "defined here");
                    }
                }

                err.emit();

                // This is the "default" function signature, used in case of error.
                // In that case, we check each argument against "error" in order to
                // set up all the node type bindings.
                (ty::Binder(self.tcx.mk_fn_sig(
                    self.err_args(arg_exprs.len()).into_iter(),
                    self.tcx.types.err,
                    false,
                    hir::Unsafety::Normal,
                    abi::Abi::Rust
                )), None)
            }
        };

        // Replace any late-bound regions that appear in the function
        // signature with region variables. We also have to
        // renormalize the associated types at this point, since they
        // previously appeared within a `Binder<>` and hence would not
        // have been normalized before.
        let fn_sig =
            self.replace_late_bound_regions_with_fresh_var(call_expr.span, infer::FnCall, &fn_sig)
                .0;
        let fn_sig = self.normalize_associated_types_in(call_expr.span, &fn_sig);

        // Call the generic checker.
        let expected_arg_tys =
            self.expected_inputs_for_expected_output(call_expr.span,
                                            expected,
                                            fn_sig.output(),
                                            fn_sig.inputs());
        self.check_argument_types(call_expr.span,
                                  call_expr.span,
                                  fn_sig.inputs(),
                                  &expected_arg_tys[..],
                                  arg_exprs,
                                  fn_sig.variadic,
                                  TupleArgumentsFlag::DontTupleArguments,
                                  def_span);

        fn_sig.output()
    }

    fn confirm_deferred_closure_call(&self,
                                     call_expr: &hir::Expr,
                                     arg_exprs: &'gcx [hir::Expr],
                                     expected: Expectation<'tcx>,
                                     fn_sig: ty::FnSig<'tcx>)
                                     -> Ty<'tcx> {
        // `fn_sig` is the *signature* of the cosure being called. We
        // don't know the full details yet (`Fn` vs `FnMut` etc), but we
        // do know the types expected for each argument and the return
        // type.

        let expected_arg_tys = self.expected_inputs_for_expected_output(call_expr.span,
                                                               expected,
                                                               fn_sig.output().clone(),
                                                               fn_sig.inputs());

        self.check_argument_types(call_expr.span,
                                  call_expr.span,
                                  fn_sig.inputs(),
                                  &expected_arg_tys,
                                  arg_exprs,
                                  fn_sig.variadic,
                                  TupleArgumentsFlag::TupleArguments,
                                  None);

        fn_sig.output()
    }

    fn confirm_overloaded_call(&self,
                               call_expr: &hir::Expr,
                               arg_exprs: &'gcx [hir::Expr],
                               expected: Expectation<'tcx>,
                               method_callee: MethodCallee<'tcx>)
                               -> Ty<'tcx> {
        let output_type = self.check_method_argument_types(call_expr.span,
                                                           call_expr.span,
                                                           Ok(method_callee),
                                                           arg_exprs,
                                                           TupleArgumentsFlag::TupleArguments,
                                                           expected);

        self.write_method_call(call_expr.hir_id, method_callee);
        output_type
    }
}

#[derive(Debug)]
pub struct DeferredCallResolution<'gcx: 'tcx, 'tcx> {
    call_expr: &'gcx hir::Expr,
    callee_expr: &'gcx hir::Expr,
    adjusted_ty: Ty<'tcx>,
    adjustments: Vec<Adjustment<'tcx>>,
    fn_sig: ty::FnSig<'tcx>,
    closure_def_id: DefId,
    closure_substs: ty::ClosureSubsts<'tcx>,
}

impl<'a, 'gcx, 'tcx> DeferredCallResolution<'gcx, 'tcx> {
    pub fn resolve(self, fcx: &FnCtxt<'a, 'gcx, 'tcx>) {
        debug!("DeferredCallResolution::resolve() {:?}", self);

        // we should not be invoked until the closure kind has been
        // determined by upvar inference
        assert!(fcx.closure_kind(self.closure_def_id, self.closure_substs).is_some());

        // We may now know enough to figure out fn vs fnmut etc.
        match fcx.try_overloaded_call_traits(self.call_expr,
                                             self.adjusted_ty) {
            Some((autoref, method_callee)) => {
                // One problem is that when we get here, we are going
                // to have a newly instantiated function signature
                // from the call trait. This has to be reconciled with
                // the older function signature we had before. In
                // principle we *should* be able to fn_sigs(), but we
                // can't because of the annoying need for a TypeTrace.
                // (This always bites me, should find a way to
                // refactor it.)
                let method_sig = method_callee.sig;

                debug!("attempt_resolution: method_callee={:?}", method_callee);

                for (method_arg_ty, self_arg_ty) in
                    method_sig.inputs().iter().skip(1).zip(self.fn_sig.inputs()) {
                    fcx.demand_eqtype(self.call_expr.span, &self_arg_ty, &method_arg_ty);
                }

                fcx.demand_eqtype(self.call_expr.span, method_sig.output(), self.fn_sig.output());

                let mut adjustments = self.adjustments;
                adjustments.extend(autoref);
                fcx.apply_adjustments(self.callee_expr, adjustments);

                fcx.write_method_call(self.call_expr.hir_id,
                                      method_callee);
            }
            None => {
                span_bug!(self.call_expr.span,
                          "failed to find an overloaded call trait for closure call");
            }
        }
    }
}
