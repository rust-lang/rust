use std::iter;

use rustc_abi::ExternAbi;
use rustc_ast::util::parser::ExprPrecedence;
use rustc_errors::{Applicability, Diag, ErrorGuaranteed, StashKey};
use rustc_hir::def::{self, CtorKind, Namespace, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir, HirId, LangItem};
use rustc_hir_analysis::autoderef::Autoderef;
use rustc_infer::infer;
use rustc_infer::traits::{Obligation, ObligationCause, ObligationCauseCode};
use rustc_middle::ty::adjustment::{
    Adjust, Adjustment, AllowTwoPhase, AutoBorrow, AutoBorrowMutability,
};
use rustc_middle::ty::{self, GenericArgsRef, Ty, TyCtxt, TypeVisitableExt};
use rustc_middle::{bug, span_bug};
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, sym};
use rustc_trait_selection::error_reporting::traits::DefIdOrName;
use rustc_trait_selection::infer::InferCtxtExt as _;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use tracing::{debug, instrument};

use super::method::MethodCallee;
use super::method::probe::ProbeScope;
use super::{Expectation, FnCtxt, TupleArgumentsFlag};
use crate::{errors, fluent_generated};

/// Checks that it is legal to call methods of the trait corresponding
/// to `trait_id` (this only cares about the trait, not the specific
/// method that is called).
pub(crate) fn check_legal_trait_for_method_call(
    tcx: TyCtxt<'_>,
    span: Span,
    receiver: Option<Span>,
    expr_span: Span,
    trait_id: DefId,
    _body_id: DefId,
) -> Result<(), ErrorGuaranteed> {
    if tcx.is_lang_item(trait_id, LangItem::Drop) {
        let sugg = if let Some(receiver) = receiver.filter(|s| !s.is_empty()) {
            errors::ExplicitDestructorCallSugg::Snippet {
                lo: expr_span.shrink_to_lo(),
                hi: receiver.shrink_to_hi().to(expr_span.shrink_to_hi()),
            }
        } else {
            errors::ExplicitDestructorCallSugg::Empty(span)
        };
        return Err(tcx.dcx().emit_err(errors::ExplicitDestructorCall { span, sugg }));
    }
    tcx.ensure_ok().coherent_trait(trait_id)
}

#[derive(Debug)]
enum CallStep<'tcx> {
    Builtin(Ty<'tcx>),
    DeferredClosure(LocalDefId, ty::FnSig<'tcx>),
    /// Call overloading when callee implements one of the Fn* traits.
    Overloaded(MethodCallee<'tcx>),
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(crate) fn check_expr_call(
        &self,
        call_expr: &'tcx hir::Expr<'tcx>,
        callee_expr: &'tcx hir::Expr<'tcx>,
        arg_exprs: &'tcx [hir::Expr<'tcx>],
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let original_callee_ty = match &callee_expr.kind {
            hir::ExprKind::Path(hir::QPath::Resolved(..) | hir::QPath::TypeRelative(..)) => self
                .check_expr_with_expectation_and_args(
                    callee_expr,
                    Expectation::NoExpectation,
                    Some((call_expr, arg_exprs)),
                ),
            _ => self.check_expr(callee_expr),
        };

        let expr_ty = self.structurally_resolve_type(call_expr.span, original_callee_ty);

        let mut autoderef = self.autoderef(callee_expr.span, expr_ty);
        let mut result = None;
        while result.is_none() && autoderef.next().is_some() {
            result = self.try_overloaded_call_step(call_expr, callee_expr, arg_exprs, &autoderef);
        }
        self.check_call_custom_abi(autoderef.final_ty(false), call_expr.span);
        self.register_predicates(autoderef.into_obligations());

        let output = match result {
            None => {
                // Check all of the arg expressions, but with no expectations
                // since we don't have a signature to compare them to.
                for arg in arg_exprs {
                    self.check_expr(arg);
                }

                if let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = &callee_expr.kind
                    && let [segment] = path.segments
                {
                    self.dcx().try_steal_modify_and_emit_err(
                        segment.ident.span,
                        StashKey::CallIntoMethod,
                        |err| {
                            // Try suggesting `foo(a)` -> `a.foo()` if possible.
                            self.suggest_call_as_method(
                                err, segment, arg_exprs, call_expr, expected,
                            );
                        },
                    );
                }

                let guar = self.report_invalid_callee(call_expr, callee_expr, expr_ty, arg_exprs);
                Ty::new_error(self.tcx, guar)
            }

            Some(CallStep::Builtin(callee_ty)) => {
                self.confirm_builtin_call(call_expr, callee_expr, callee_ty, arg_exprs, expected)
            }

            Some(CallStep::DeferredClosure(def_id, fn_sig)) => {
                self.confirm_deferred_closure_call(call_expr, arg_exprs, expected, def_id, fn_sig)
            }

            Some(CallStep::Overloaded(method_callee)) => {
                self.confirm_overloaded_call(call_expr, arg_exprs, expected, method_callee)
            }
        };

        // we must check that return type of called functions is WF:
        self.register_wf_obligation(
            output.into(),
            call_expr.span,
            ObligationCauseCode::WellFormed(None),
        );

        output
    }

    /// Functions of type `extern "custom" fn(/* ... */)` cannot be called using `ExprKind::Call`.
    ///
    /// These functions have a calling convention that is unknown to rust, hence it cannot generate
    /// code for the call. The only way to execute such a function is via inline assembly.
    fn check_call_custom_abi(&self, callee_ty: Ty<'tcx>, span: Span) {
        let abi = match callee_ty.kind() {
            ty::FnDef(def_id, _) => self.tcx.fn_sig(def_id).skip_binder().skip_binder().abi,
            ty::FnPtr(_, header) => header.abi,
            _ => return,
        };

        if let ExternAbi::Custom = abi {
            self.tcx.dcx().emit_err(errors::AbiCustomCall { span });
        }
    }

    #[instrument(level = "debug", skip(self, call_expr, callee_expr, arg_exprs, autoderef), ret)]
    fn try_overloaded_call_step(
        &self,
        call_expr: &'tcx hir::Expr<'tcx>,
        callee_expr: &'tcx hir::Expr<'tcx>,
        arg_exprs: &'tcx [hir::Expr<'tcx>],
        autoderef: &Autoderef<'a, 'tcx>,
    ) -> Option<CallStep<'tcx>> {
        let adjusted_ty =
            self.structurally_resolve_type(autoderef.span(), autoderef.final_ty(false));

        // If the callee is a bare function or a closure, then we're all set.
        match *adjusted_ty.kind() {
            ty::FnDef(..) | ty::FnPtr(..) => {
                let adjustments = self.adjust_steps(autoderef);
                self.apply_adjustments(callee_expr, adjustments);
                return Some(CallStep::Builtin(adjusted_ty));
            }

            // Check whether this is a call to a closure where we
            // haven't yet decided on whether the closure is fn vs
            // fnmut vs fnonce. If so, we have to defer further processing.
            ty::Closure(def_id, args) if self.closure_kind(adjusted_ty).is_none() => {
                let def_id = def_id.expect_local();
                let closure_sig = args.as_closure().sig();
                let closure_sig = self.instantiate_binder_with_fresh_vars(
                    call_expr.span,
                    infer::FnCall,
                    closure_sig,
                );
                let adjustments = self.adjust_steps(autoderef);
                self.record_deferred_call_resolution(
                    def_id,
                    DeferredCallResolution {
                        call_expr,
                        callee_expr,
                        closure_ty: adjusted_ty,
                        adjustments,
                        fn_sig: closure_sig,
                    },
                );
                return Some(CallStep::DeferredClosure(def_id, closure_sig));
            }

            // When calling a `CoroutineClosure` that is local to the body, we will
            // not know what its `closure_kind` is yet. Instead, just fill in the
            // signature with an infer var for the `tupled_upvars_ty` of the coroutine,
            // and record a deferred call resolution which will constrain that var
            // as part of `AsyncFn*` trait confirmation.
            ty::CoroutineClosure(def_id, args) if self.closure_kind(adjusted_ty).is_none() => {
                let def_id = def_id.expect_local();
                let closure_args = args.as_coroutine_closure();
                let coroutine_closure_sig = self.instantiate_binder_with_fresh_vars(
                    call_expr.span,
                    infer::FnCall,
                    closure_args.coroutine_closure_sig(),
                );
                let tupled_upvars_ty = self.next_ty_var(callee_expr.span);
                // We may actually receive a coroutine back whose kind is different
                // from the closure that this dispatched from. This is because when
                // we have no captures, we automatically implement `FnOnce`. This
                // impl forces the closure kind to `FnOnce` i.e. `u8`.
                let kind_ty = self.next_ty_var(callee_expr.span);
                let call_sig = self.tcx.mk_fn_sig(
                    [coroutine_closure_sig.tupled_inputs_ty],
                    coroutine_closure_sig.to_coroutine(
                        self.tcx,
                        closure_args.parent_args(),
                        kind_ty,
                        self.tcx.coroutine_for_closure(def_id),
                        tupled_upvars_ty,
                    ),
                    coroutine_closure_sig.c_variadic,
                    coroutine_closure_sig.safety,
                    coroutine_closure_sig.abi,
                );
                let adjustments = self.adjust_steps(autoderef);
                self.record_deferred_call_resolution(
                    def_id,
                    DeferredCallResolution {
                        call_expr,
                        callee_expr,
                        closure_ty: adjusted_ty,
                        adjustments,
                        fn_sig: call_sig,
                    },
                );
                return Some(CallStep::DeferredClosure(def_id, call_sig));
            }

            // Hack: we know that there are traits implementing Fn for &F
            // where F:Fn and so forth. In the particular case of types
            // like `f: &mut FnMut()`, if there is a call `f()`, we would
            // normally translate to `FnMut::call_mut(&mut f, ())`, but
            // that winds up potentially requiring the user to mark their
            // variable as `mut` which feels unnecessary and unexpected.
            //
            //     fn foo(f: &mut impl FnMut()) { f() }
            //            ^ without this hack `f` would have to be declared as mutable
            //
            // The simplest fix by far is to just ignore this case and deref again,
            // so we wind up with `FnMut::call_mut(&mut *f, ())`.
            ty::Ref(..) if autoderef.step_count() == 0 => {
                return None;
            }

            ty::Error(_) => {
                return None;
            }

            _ => {}
        }

        // Now, we look for the implementation of a Fn trait on the object's type.
        // We first do it with the explicit instruction to look for an impl of
        // `Fn<Tuple>`, with the tuple `Tuple` having an arity corresponding
        // to the number of call parameters.
        // If that fails (or_else branch), we try again without specifying the
        // shape of the tuple (hence the None). This allows to detect an Fn trait
        // is implemented, and use this information for diagnostic.
        self.try_overloaded_call_traits(call_expr, adjusted_ty, Some(arg_exprs))
            .or_else(|| self.try_overloaded_call_traits(call_expr, adjusted_ty, None))
            .map(|(autoref, method)| {
                let mut adjustments = self.adjust_steps(autoderef);
                adjustments.extend(autoref);
                self.apply_adjustments(callee_expr, adjustments);
                CallStep::Overloaded(method)
            })
    }

    fn try_overloaded_call_traits(
        &self,
        call_expr: &hir::Expr<'_>,
        adjusted_ty: Ty<'tcx>,
        opt_arg_exprs: Option<&'tcx [hir::Expr<'tcx>]>,
    ) -> Option<(Option<Adjustment<'tcx>>, MethodCallee<'tcx>)> {
        // HACK(async_closures): For async closures, prefer `AsyncFn*`
        // over `Fn*`, since all async closures implement `FnOnce`, but
        // choosing that over `AsyncFn`/`AsyncFnMut` would be more restrictive.
        // For other callables, just prefer `Fn*` for perf reasons.
        //
        // The order of trait choices here is not that big of a deal,
        // since it just guides inference (and our choice of autoref).
        // Though in the future, I'd like typeck to choose:
        // `Fn > AsyncFn > FnMut > AsyncFnMut > FnOnce > AsyncFnOnce`
        // ...or *ideally*, we just have `LendingFn`/`LendingFnMut`, which
        // would naturally unify these two trait hierarchies in the most
        // general way.
        let call_trait_choices = if self.shallow_resolve(adjusted_ty).is_coroutine_closure() {
            [
                (self.tcx.lang_items().async_fn_trait(), sym::async_call, true),
                (self.tcx.lang_items().async_fn_mut_trait(), sym::async_call_mut, true),
                (self.tcx.lang_items().async_fn_once_trait(), sym::async_call_once, false),
                (self.tcx.lang_items().fn_trait(), sym::call, true),
                (self.tcx.lang_items().fn_mut_trait(), sym::call_mut, true),
                (self.tcx.lang_items().fn_once_trait(), sym::call_once, false),
            ]
        } else {
            [
                (self.tcx.lang_items().fn_trait(), sym::call, true),
                (self.tcx.lang_items().fn_mut_trait(), sym::call_mut, true),
                (self.tcx.lang_items().fn_once_trait(), sym::call_once, false),
                (self.tcx.lang_items().async_fn_trait(), sym::async_call, true),
                (self.tcx.lang_items().async_fn_mut_trait(), sym::async_call_mut, true),
                (self.tcx.lang_items().async_fn_once_trait(), sym::async_call_once, false),
            ]
        };

        // Try the options that are least restrictive on the caller first.
        for (opt_trait_def_id, method_name, borrow) in call_trait_choices {
            let Some(trait_def_id) = opt_trait_def_id else { continue };

            let opt_input_type = opt_arg_exprs.map(|arg_exprs| {
                Ty::new_tup_from_iter(self.tcx, arg_exprs.iter().map(|e| self.next_ty_var(e.span)))
            });

            if let Some(ok) = self.lookup_method_for_operator(
                self.misc(call_expr.span),
                method_name,
                trait_def_id,
                adjusted_ty,
                opt_input_type,
            ) {
                let method = self.register_infer_ok_obligations(ok);
                let mut autoref = None;
                if borrow {
                    // Check for &self vs &mut self in the method signature. Since this is either
                    // the Fn or FnMut trait, it should be one of those.
                    let ty::Ref(_, _, mutbl) = method.sig.inputs()[0].kind() else {
                        bug!("Expected `FnMut`/`Fn` to take receiver by-ref/by-mut")
                    };

                    // For initial two-phase borrow
                    // deployment, conservatively omit
                    // overloaded function call ops.
                    let mutbl = AutoBorrowMutability::new(*mutbl, AllowTwoPhase::No);

                    autoref = Some(Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::Ref(mutbl)),
                        target: method.sig.inputs()[0],
                    });
                }

                return Some((autoref, method));
            }
        }

        None
    }

    /// Give appropriate suggestion when encountering `||{/* not callable */}()`, where the
    /// likely intention is to call the closure, suggest `(||{})()`. (#55851)
    fn identify_bad_closure_def_and_call(
        &self,
        err: &mut Diag<'_>,
        hir_id: hir::HirId,
        callee_node: &hir::ExprKind<'_>,
        callee_span: Span,
    ) {
        let hir::ExprKind::Block(..) = callee_node else {
            // Only calls on blocks suggested here.
            return;
        };

        let fn_decl_span = if let hir::Node::Expr(&hir::Expr {
            kind: hir::ExprKind::Closure(&hir::Closure { fn_decl_span, .. }),
            ..
        }) = self.tcx.parent_hir_node(hir_id)
        {
            fn_decl_span
        } else if let Some((
            _,
            hir::Node::Expr(&hir::Expr {
                hir_id: parent_hir_id,
                kind:
                    hir::ExprKind::Closure(&hir::Closure {
                        kind:
                            hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                                hir::CoroutineDesugaring::Async,
                                hir::CoroutineSource::Closure,
                            )),
                        ..
                    }),
                ..
            }),
        )) = self.tcx.hir_parent_iter(hir_id).nth(3)
        {
            // Actually need to unwrap one more layer of HIR to get to
            // the _real_ closure...
            if let hir::Node::Expr(&hir::Expr {
                kind: hir::ExprKind::Closure(&hir::Closure { fn_decl_span, .. }),
                ..
            }) = self.tcx.parent_hir_node(parent_hir_id)
            {
                fn_decl_span
            } else {
                return;
            }
        } else {
            return;
        };

        let start = fn_decl_span.shrink_to_lo();
        let end = callee_span.shrink_to_hi();
        err.multipart_suggestion(
            "if you meant to create this closure and immediately call it, surround the \
                closure with parentheses",
            vec![(start, "(".to_string()), (end, ")".to_string())],
            Applicability::MaybeIncorrect,
        );
    }

    /// Give appropriate suggestion when encountering `[("a", 0) ("b", 1)]`, where the
    /// likely intention is to create an array containing tuples.
    fn maybe_suggest_bad_array_definition(
        &self,
        err: &mut Diag<'_>,
        call_expr: &'tcx hir::Expr<'tcx>,
        callee_expr: &'tcx hir::Expr<'tcx>,
    ) -> bool {
        let parent_node = self.tcx.parent_hir_node(call_expr.hir_id);
        if let (
            hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Array(_), .. }),
            hir::ExprKind::Tup(exp),
            hir::ExprKind::Call(_, args),
        ) = (parent_node, &callee_expr.kind, &call_expr.kind)
            && args.len() == exp.len()
        {
            let start = callee_expr.span.shrink_to_hi();
            err.span_suggestion(
                start,
                "consider separating array elements with a comma",
                ",",
                Applicability::MaybeIncorrect,
            );
            return true;
        }
        false
    }

    fn confirm_builtin_call(
        &self,
        call_expr: &'tcx hir::Expr<'tcx>,
        callee_expr: &'tcx hir::Expr<'tcx>,
        callee_ty: Ty<'tcx>,
        arg_exprs: &'tcx [hir::Expr<'tcx>],
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let (fn_sig, def_id) = match *callee_ty.kind() {
            ty::FnDef(def_id, args) => {
                self.enforce_context_effects(Some(call_expr.hir_id), call_expr.span, def_id, args);
                let fn_sig = self.tcx.fn_sig(def_id).instantiate(self.tcx, args);

                // Unit testing: function items annotated with
                // `#[rustc_evaluate_where_clauses]` trigger special output
                // to let us test the trait evaluation system.
                // Untranslatable diagnostics are okay for rustc internals
                #[allow(rustc::untranslatable_diagnostic)]
                #[allow(rustc::diagnostic_outside_of_impl)]
                if self.tcx.has_attr(def_id, sym::rustc_evaluate_where_clauses) {
                    let predicates = self.tcx.predicates_of(def_id);
                    let predicates = predicates.instantiate(self.tcx, args);
                    for (predicate, predicate_span) in predicates {
                        let obligation = Obligation::new(
                            self.tcx,
                            ObligationCause::dummy_with_span(callee_expr.span),
                            self.param_env,
                            predicate,
                        );
                        let result = self.evaluate_obligation(&obligation);
                        self.dcx()
                            .struct_span_err(
                                callee_expr.span,
                                format!("evaluate({predicate:?}) = {result:?}"),
                            )
                            .with_span_label(predicate_span, "predicate")
                            .emit();
                    }
                }
                (fn_sig, Some(def_id))
            }

            // FIXME(const_trait_impl): these arms should error because we can't enforce them
            ty::FnPtr(sig_tys, hdr) => (sig_tys.with(hdr), None),

            _ => unreachable!(),
        };

        // Replace any late-bound regions that appear in the function
        // signature with region variables. We also have to
        // renormalize the associated types at this point, since they
        // previously appeared within a `Binder<>` and hence would not
        // have been normalized before.
        let fn_sig = self.instantiate_binder_with_fresh_vars(call_expr.span, infer::FnCall, fn_sig);
        let fn_sig = self.normalize(call_expr.span, fn_sig);

        self.check_argument_types(
            call_expr.span,
            call_expr,
            fn_sig.inputs(),
            fn_sig.output(),
            expected,
            arg_exprs,
            fn_sig.c_variadic,
            TupleArgumentsFlag::DontTupleArguments,
            def_id,
        );

        if fn_sig.abi == rustc_abi::ExternAbi::RustCall {
            let sp = arg_exprs.last().map_or(call_expr.span, |expr| expr.span);
            if let Some(ty) = fn_sig.inputs().last().copied() {
                self.register_bound(
                    ty,
                    self.tcx.require_lang_item(hir::LangItem::Tuple, sp),
                    self.cause(sp, ObligationCauseCode::RustCall),
                );
                self.require_type_is_sized(ty, sp, ObligationCauseCode::RustCall);
            } else {
                self.dcx().emit_err(errors::RustCallIncorrectArgs { span: sp });
            }
        }

        if let Some(def_id) = def_id
            && self.tcx.def_kind(def_id) == hir::def::DefKind::Fn
            && self.tcx.is_intrinsic(def_id, sym::const_eval_select)
        {
            let fn_sig = self.resolve_vars_if_possible(fn_sig);
            for idx in 0..=1 {
                let arg_ty = fn_sig.inputs()[idx + 1];
                let span = arg_exprs.get(idx + 1).map_or(call_expr.span, |arg| arg.span);
                // Check that second and third argument of `const_eval_select` must be `FnDef`, and additionally that
                // the second argument must be `const fn`. The first argument must be a tuple, but this is already expressed
                // in the function signature (`F: FnOnce<ARG>`), so I did not bother to add another check here.
                //
                // This check is here because there is currently no way to express a trait bound for `FnDef` types only.
                if let ty::FnDef(def_id, _args) = *arg_ty.kind() {
                    if idx == 0 && !self.tcx.is_const_fn(def_id) {
                        self.dcx().emit_err(errors::ConstSelectMustBeConst { span });
                    }
                } else {
                    self.dcx().emit_err(errors::ConstSelectMustBeFn { span, ty: arg_ty });
                }
            }
        }

        fn_sig.output()
    }

    /// Attempts to reinterpret `method(rcvr, args...)` as `rcvr.method(args...)`
    /// and suggesting the fix if the method probe is successful.
    fn suggest_call_as_method(
        &self,
        diag: &mut Diag<'_>,
        segment: &'tcx hir::PathSegment<'tcx>,
        arg_exprs: &'tcx [hir::Expr<'tcx>],
        call_expr: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
    ) {
        if let [callee_expr, rest @ ..] = arg_exprs {
            let Some(callee_ty) = self.typeck_results.borrow().expr_ty_adjusted_opt(callee_expr)
            else {
                return;
            };

            // First, do a probe with `IsSuggestion(true)` to avoid emitting
            // any strange errors. If it's successful, then we'll do a true
            // method lookup.
            let Ok(pick) = self.lookup_probe_for_diagnostic(
                segment.ident,
                callee_ty,
                call_expr,
                // We didn't record the in scope traits during late resolution
                // so we need to probe AllTraits unfortunately
                ProbeScope::AllTraits,
                expected.only_has_type(self),
            ) else {
                return;
            };

            let pick = self.confirm_method_for_diagnostic(
                call_expr.span,
                callee_expr,
                call_expr,
                callee_ty,
                &pick,
                segment,
            );
            if pick.illegal_sized_bound.is_some() {
                return;
            }

            let Some(callee_expr_span) = callee_expr.span.find_ancestor_inside(call_expr.span)
            else {
                return;
            };
            let up_to_rcvr_span = segment.ident.span.until(callee_expr_span);
            let rest_span = callee_expr_span.shrink_to_hi().to(call_expr.span.shrink_to_hi());
            let rest_snippet = if let Some(first) = rest.first() {
                self.tcx
                    .sess
                    .source_map()
                    .span_to_snippet(first.span.to(call_expr.span.shrink_to_hi()))
            } else {
                Ok(")".to_string())
            };

            if let Ok(rest_snippet) = rest_snippet {
                let sugg = if callee_expr.precedence() >= ExprPrecedence::Unambiguous {
                    vec![
                        (up_to_rcvr_span, "".to_string()),
                        (rest_span, format!(".{}({rest_snippet}", segment.ident)),
                    ]
                } else {
                    vec![
                        (up_to_rcvr_span, "(".to_string()),
                        (rest_span, format!(").{}({rest_snippet}", segment.ident)),
                    ]
                };
                let self_ty = self.resolve_vars_if_possible(pick.callee.sig.inputs()[0]);
                diag.multipart_suggestion(
                    format!(
                        "use the `.` operator to call the method `{}{}` on `{self_ty}`",
                        self.tcx
                            .associated_item(pick.callee.def_id)
                            .trait_container(self.tcx)
                            .map_or_else(
                                || String::new(),
                                |trait_def_id| self.tcx.def_path_str(trait_def_id) + "::"
                            ),
                        segment.ident
                    ),
                    sugg,
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }

    fn report_invalid_callee(
        &self,
        call_expr: &'tcx hir::Expr<'tcx>,
        callee_expr: &'tcx hir::Expr<'tcx>,
        callee_ty: Ty<'tcx>,
        arg_exprs: &'tcx [hir::Expr<'tcx>],
    ) -> ErrorGuaranteed {
        // Callee probe fails when APIT references errors, so suppress those
        // errors here.
        if let Some((_, _, args)) = self.extract_callable_info(callee_ty)
            && let Err(err) = args.error_reported()
        {
            return err;
        }

        let mut unit_variant = None;
        if let hir::ExprKind::Path(qpath) = &callee_expr.kind
            && let Res::Def(def::DefKind::Ctor(kind, CtorKind::Const), _)
                = self.typeck_results.borrow().qpath_res(qpath, callee_expr.hir_id)
            // Only suggest removing parens if there are no arguments
            && arg_exprs.is_empty()
            && call_expr.span.contains(callee_expr.span)
        {
            let descr = match kind {
                def::CtorOf::Struct => "struct",
                def::CtorOf::Variant => "enum variant",
            };
            let removal_span = callee_expr.span.shrink_to_hi().to(call_expr.span.shrink_to_hi());
            unit_variant =
                Some((removal_span, descr, rustc_hir_pretty::qpath_to_string(&self.tcx, qpath)));
        }

        let callee_ty = self.resolve_vars_if_possible(callee_ty);
        let mut path = None;
        let mut err = self.dcx().create_err(errors::InvalidCallee {
            span: callee_expr.span,
            ty: callee_ty,
            found: match &unit_variant {
                Some((_, kind, path)) => format!("{kind} `{path}`"),
                None => format!("`{}`", self.tcx.short_string(callee_ty, &mut path)),
            },
        });
        *err.long_ty_path() = path;
        if callee_ty.references_error() {
            err.downgrade_to_delayed_bug();
        }

        self.identify_bad_closure_def_and_call(
            &mut err,
            call_expr.hir_id,
            &callee_expr.kind,
            callee_expr.span,
        );

        if let Some((removal_span, kind, path)) = &unit_variant {
            err.span_suggestion_verbose(
                *removal_span,
                format!(
                    "`{path}` is a unit {kind}, and does not take parentheses to be constructed",
                ),
                "",
                Applicability::MachineApplicable,
            );
        }

        if let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = callee_expr.kind
            && let Res::Local(_) = path.res
            && let [segment] = &path.segments
        {
            for id in self.tcx.hir_free_items() {
                if let Some(node) = self.tcx.hir_get_if_local(id.owner_id.into())
                    && let hir::Node::Item(item) = node
                    && let hir::ItemKind::Fn { ident, .. } = item.kind
                    && ident.name == segment.ident.name
                {
                    err.span_label(
                        self.tcx.def_span(id.owner_id),
                        "this function of the same name is available here, but it's shadowed by \
                         the local binding",
                    );
                }
            }
        }

        let mut inner_callee_path = None;
        let def = match callee_expr.kind {
            hir::ExprKind::Path(ref qpath) => {
                self.typeck_results.borrow().qpath_res(qpath, callee_expr.hir_id)
            }
            hir::ExprKind::Call(inner_callee, _) => {
                if let hir::ExprKind::Path(ref inner_qpath) = inner_callee.kind {
                    inner_callee_path = Some(inner_qpath);
                    self.typeck_results.borrow().qpath_res(inner_qpath, inner_callee.hir_id)
                } else {
                    Res::Err
                }
            }
            _ => Res::Err,
        };

        if !self.maybe_suggest_bad_array_definition(&mut err, call_expr, callee_expr) {
            // If the call spans more than one line and the callee kind is
            // itself another `ExprCall`, that's a clue that we might just be
            // missing a semicolon (#51055, #106515).
            let call_is_multiline = self
                .tcx
                .sess
                .source_map()
                .is_multiline(call_expr.span.with_lo(callee_expr.span.hi()))
                && call_expr.span.eq_ctxt(callee_expr.span);
            if call_is_multiline {
                err.span_suggestion(
                    callee_expr.span.shrink_to_hi(),
                    "consider using a semicolon here to finish the statement",
                    ";",
                    Applicability::MaybeIncorrect,
                );
            }
            if let Some((maybe_def, output_ty, _)) = self.extract_callable_info(callee_ty)
                && !self.type_is_sized_modulo_regions(self.param_env, output_ty)
            {
                let descr = match maybe_def {
                    DefIdOrName::DefId(def_id) => self.tcx.def_descr(def_id),
                    DefIdOrName::Name(name) => name,
                };
                err.span_label(
                    callee_expr.span,
                    format!("this {descr} returns an unsized value `{output_ty}`, so it cannot be called")
                );
                if let DefIdOrName::DefId(def_id) = maybe_def
                    && let Some(def_span) = self.tcx.hir_span_if_local(def_id)
                {
                    err.span_label(def_span, "the callable type is defined here");
                }
            } else {
                err.span_label(call_expr.span, "call expression requires function");
            }
        }

        if let Some(span) = self.tcx.hir_res_span(def) {
            let callee_ty = callee_ty.to_string();
            let label = match (unit_variant, inner_callee_path) {
                (Some((_, kind, path)), _) => {
                    err.arg("kind", kind);
                    err.arg("path", path);
                    Some(fluent_generated::hir_typeck_invalid_defined_kind)
                }
                (_, Some(hir::QPath::Resolved(_, path))) => {
                    self.tcx.sess.source_map().span_to_snippet(path.span).ok().map(|p| {
                        err.arg("func", p);
                        fluent_generated::hir_typeck_invalid_fn_defined
                    })
                }
                _ => {
                    match def {
                        // Emit a different diagnostic for local variables, as they are not
                        // type definitions themselves, but rather variables *of* that type.
                        Res::Local(hir_id) => {
                            err.arg("local_name", self.tcx.hir_name(hir_id));
                            Some(fluent_generated::hir_typeck_invalid_local)
                        }
                        Res::Def(kind, def_id) if kind.ns() == Some(Namespace::ValueNS) => {
                            err.arg("path", self.tcx.def_path_str(def_id));
                            Some(fluent_generated::hir_typeck_invalid_defined)
                        }
                        _ => {
                            err.arg("path", callee_ty);
                            Some(fluent_generated::hir_typeck_invalid_defined)
                        }
                    }
                }
            };
            if let Some(label) = label {
                err.span_label(span, label);
            }
        }
        err.emit()
    }

    fn confirm_deferred_closure_call(
        &self,
        call_expr: &'tcx hir::Expr<'tcx>,
        arg_exprs: &'tcx [hir::Expr<'tcx>],
        expected: Expectation<'tcx>,
        closure_def_id: LocalDefId,
        fn_sig: ty::FnSig<'tcx>,
    ) -> Ty<'tcx> {
        // `fn_sig` is the *signature* of the closure being called. We
        // don't know the full details yet (`Fn` vs `FnMut` etc), but we
        // do know the types expected for each argument and the return
        // type.
        self.check_argument_types(
            call_expr.span,
            call_expr,
            fn_sig.inputs(),
            fn_sig.output(),
            expected,
            arg_exprs,
            fn_sig.c_variadic,
            TupleArgumentsFlag::TupleArguments,
            Some(closure_def_id.to_def_id()),
        );

        fn_sig.output()
    }

    #[tracing::instrument(level = "debug", skip(self, span))]
    pub(super) fn enforce_context_effects(
        &self,
        call_hir_id: Option<HirId>,
        span: Span,
        callee_did: DefId,
        callee_args: GenericArgsRef<'tcx>,
    ) {
        // FIXME(const_trait_impl): We should be enforcing these effects unconditionally.
        // This can be done as soon as we convert the standard library back to
        // using const traits, since if we were to enforce these conditions now,
        // we'd fail on basically every builtin trait call (i.e. `1 + 2`).
        if !self.tcx.features().const_trait_impl() {
            return;
        }

        // If we have `rustc_do_not_const_check`, do not check `~const` bounds.
        if self.tcx.has_attr(self.body_id, sym::rustc_do_not_const_check) {
            return;
        }

        let host = match self.tcx.hir_body_const_context(self.body_id) {
            Some(hir::ConstContext::Const { .. } | hir::ConstContext::Static(_)) => {
                ty::BoundConstness::Const
            }
            Some(hir::ConstContext::ConstFn) => ty::BoundConstness::Maybe,
            None => return,
        };

        // FIXME(const_trait_impl): Should this be `is_const_fn_raw`? It depends on if we move
        // const stability checking here too, I guess.
        if self.tcx.is_conditionally_const(callee_did) {
            let q = self.tcx.const_conditions(callee_did);
            // FIXME(const_trait_impl): Use this span with a better cause code.
            for (idx, (cond, pred_span)) in
                q.instantiate(self.tcx, callee_args).into_iter().enumerate()
            {
                let cause = self.cause(
                    span,
                    if let Some(hir_id) = call_hir_id {
                        ObligationCauseCode::HostEffectInExpr(callee_did, pred_span, hir_id, idx)
                    } else {
                        ObligationCauseCode::WhereClause(callee_did, pred_span)
                    },
                );
                self.register_predicate(Obligation::new(
                    self.tcx,
                    cause,
                    self.param_env,
                    cond.to_host_effect_clause(self.tcx, host),
                ));
            }
        } else {
            // FIXME(const_trait_impl): This should eventually be caught here.
            // For now, though, we defer some const checking to MIR.
        }
    }

    fn confirm_overloaded_call(
        &self,
        call_expr: &'tcx hir::Expr<'tcx>,
        arg_exprs: &'tcx [hir::Expr<'tcx>],
        expected: Expectation<'tcx>,
        method: MethodCallee<'tcx>,
    ) -> Ty<'tcx> {
        self.check_argument_types(
            call_expr.span,
            call_expr,
            &method.sig.inputs()[1..],
            method.sig.output(),
            expected,
            arg_exprs,
            method.sig.c_variadic,
            TupleArgumentsFlag::TupleArguments,
            Some(method.def_id),
        );

        self.write_method_call_and_enforce_effects(call_expr.hir_id, call_expr.span, method);

        method.sig.output()
    }
}

#[derive(Debug)]
pub(crate) struct DeferredCallResolution<'tcx> {
    call_expr: &'tcx hir::Expr<'tcx>,
    callee_expr: &'tcx hir::Expr<'tcx>,
    closure_ty: Ty<'tcx>,
    adjustments: Vec<Adjustment<'tcx>>,
    fn_sig: ty::FnSig<'tcx>,
}

impl<'a, 'tcx> DeferredCallResolution<'tcx> {
    pub(crate) fn resolve(self, fcx: &FnCtxt<'a, 'tcx>) {
        debug!("DeferredCallResolution::resolve() {:?}", self);

        // we should not be invoked until the closure kind has been
        // determined by upvar inference
        assert!(fcx.closure_kind(self.closure_ty).is_some());

        // We may now know enough to figure out fn vs fnmut etc.
        match fcx.try_overloaded_call_traits(self.call_expr, self.closure_ty, None) {
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
                    iter::zip(method_sig.inputs().iter().skip(1), self.fn_sig.inputs())
                {
                    fcx.demand_eqtype(self.call_expr.span, *self_arg_ty, *method_arg_ty);
                }

                fcx.demand_eqtype(self.call_expr.span, method_sig.output(), self.fn_sig.output());

                let mut adjustments = self.adjustments;
                adjustments.extend(autoref);
                fcx.apply_adjustments(self.callee_expr, adjustments);

                fcx.write_method_call_and_enforce_effects(
                    self.call_expr.hir_id,
                    self.call_expr.span,
                    method_callee,
                );
            }
            None => {
                span_bug!(
                    self.call_expr.span,
                    "Expected to find a suitable `Fn`/`FnMut`/`FnOnce` implementation for `{}`",
                    self.closure_ty
                )
            }
        }
    }
}
