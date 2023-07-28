use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::higher::VecArgs;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item};
use clippy_utils::usage::local_used_after_expr;
use clippy_utils::{higher, is_adjusted, path_to_local, path_to_local_id};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{Closure, Expr, ExprKind, Param, PatKind, Unsafety};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow};
use rustc_middle::ty::binding::BindingMode;
use rustc_middle::ty::{self, EarlyBinder, GenericArgsRef, Ty, TypeVisitableExt};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for closures which just call another function where
    /// the function can be called directly. `unsafe` functions or calls where types
    /// get adjusted are ignored.
    ///
    /// ### Why is this bad?
    /// Needlessly creating a closure adds code for no benefit
    /// and gives the optimizer more work.
    ///
    /// ### Known problems
    /// If creating the closure inside the closure has a side-
    /// effect then moving the closure creation out will change when that side-
    /// effect runs.
    /// See [#1439](https://github.com/rust-lang/rust-clippy/issues/1439) for more details.
    ///
    /// ### Example
    /// ```rust,ignore
    /// xs.map(|x| foo(x))
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// // where `foo(_)` is a plain function that takes the exact argument type of `x`.
    /// xs.map(foo)
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub REDUNDANT_CLOSURE,
    style,
    "redundant closures, i.e., `|a| foo(a)` (which can be written as just `foo`)"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for closures which only invoke a method on the closure
    /// argument and can be replaced by referencing the method directly.
    ///
    /// ### Why is this bad?
    /// It's unnecessary to create the closure.
    ///
    /// ### Example
    /// ```rust,ignore
    /// Some('a').map(|s| s.to_uppercase());
    /// ```
    /// may be rewritten as
    /// ```rust,ignore
    /// Some('a').map(char::to_uppercase);
    /// ```
    #[clippy::version = "1.35.0"]
    pub REDUNDANT_CLOSURE_FOR_METHOD_CALLS,
    pedantic,
    "redundant closures for method calls"
}

declare_lint_pass!(EtaReduction => [REDUNDANT_CLOSURE, REDUNDANT_CLOSURE_FOR_METHOD_CALLS]);

impl<'tcx> LateLintPass<'tcx> for EtaReduction {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }
        let body = match expr.kind {
            ExprKind::Closure(&Closure { body, .. }) => cx.tcx.hir().body(body),
            _ => return,
        };
        if body.value.span.from_expansion() {
            if body.params.is_empty() {
                if let Some(VecArgs::Vec(&[])) = higher::VecArgs::hir(cx, body.value) {
                    // replace `|| vec![]` with `Vec::new`
                    span_lint_and_sugg(
                        cx,
                        REDUNDANT_CLOSURE,
                        expr.span,
                        "redundant closure",
                        "replace the closure with `Vec::new`",
                        "std::vec::Vec::new".into(),
                        Applicability::MachineApplicable,
                    );
                }
            }
            // skip `foo(|| macro!())`
            return;
        }

        let closure_ty = cx.typeck_results().expr_ty(expr);

        if_chain!(
            if !is_adjusted(cx, body.value);
            if let ExprKind::Call(callee, args) = body.value.kind;
            if let ExprKind::Path(_) = callee.kind;
            if check_inputs(cx, body.params, None, args);
            let callee_ty = cx.typeck_results().expr_ty_adjusted(callee);
            let call_ty = cx.typeck_results().type_dependent_def_id(body.value.hir_id)
                .map_or(callee_ty, |id| cx.tcx.type_of(id).instantiate_identity());
            if check_sig(cx, closure_ty, call_ty);
            let args = cx.typeck_results().node_args(callee.hir_id);
            // This fixes some false positives that I don't entirely understand
            if args.is_empty() || !cx.typeck_results().expr_ty(expr).has_late_bound_regions();
            // A type param function ref like `T::f` is not 'static, however
            // it is if cast like `T::f as fn()`. This seems like a rustc bug.
            if !args.types().any(|t| matches!(t.kind(), ty::Param(_)));
            let callee_ty_unadjusted = cx.typeck_results().expr_ty(callee).peel_refs();
            if !is_type_diagnostic_item(cx, callee_ty_unadjusted, sym::Arc);
            if !is_type_diagnostic_item(cx, callee_ty_unadjusted, sym::Rc);
            if let ty::Closure(_, args) = *closure_ty.kind();
            // Don't lint if this is an inclusive range expression.
            // They desugar to a call to `RangeInclusiveNew` which would have odd suggestions. (#10684)
            if !matches!(higher::Range::hir(body.value), Some(higher::Range {
                start: Some(_),
                end: Some(_),
                limits: rustc_ast::RangeLimits::Closed
            }));
            then {
                span_lint_and_then(cx, REDUNDANT_CLOSURE, expr.span, "redundant closure", |diag| {
                    if let Some(mut snippet) = snippet_opt(cx, callee.span) {
                        if let Some(fn_mut_id) = cx.tcx.lang_items().fn_mut_trait()
                            && let args = cx.tcx.erase_late_bound_regions(args.as_closure().sig()).inputs()
                            && implements_trait(
                                   cx,
                                   callee_ty.peel_refs(),
                                   fn_mut_id,
                                   &args.iter().copied().map(Into::into).collect::<Vec<_>>(),
                               )
                            && path_to_local(callee).map_or(false, |l| local_used_after_expr(cx, l, expr))
                        {
                                // Mutable closure is used after current expr; we cannot consume it.
                                snippet = format!("&mut {snippet}");
                        }

                        diag.span_suggestion(
                            expr.span,
                            "replace the closure with the function itself",
                            snippet,
                            Applicability::MachineApplicable,
                        );
                    }
                });
            }
        );

        if_chain!(
            if !is_adjusted(cx, body.value);
            if let ExprKind::MethodCall(path, receiver, args, _) = body.value.kind;
            if check_inputs(cx, body.params, Some(receiver), args);
            let method_def_id = cx.typeck_results().type_dependent_def_id(body.value.hir_id).unwrap();
            let args = cx.typeck_results().node_args(body.value.hir_id);
            let call_ty = cx.tcx.type_of(method_def_id).instantiate(cx.tcx, args);
            if check_sig(cx, closure_ty, call_ty);
            then {
                span_lint_and_then(cx, REDUNDANT_CLOSURE_FOR_METHOD_CALLS, expr.span, "redundant closure", |diag| {
                    let name = get_ufcs_type_name(cx, method_def_id, args);
                    diag.span_suggestion(
                        expr.span,
                        "replace the closure with the method itself",
                        format!("{name}::{}", path.ident.name),
                        Applicability::MachineApplicable,
                    );
                })
            }
        );
    }
}

fn check_inputs(
    cx: &LateContext<'_>,
    params: &[Param<'_>],
    receiver: Option<&Expr<'_>>,
    call_args: &[Expr<'_>],
) -> bool {
    if receiver.map_or(params.len() != call_args.len(), |_| params.len() != call_args.len() + 1) {
        return false;
    }
    let binding_modes = cx.typeck_results().pat_binding_modes();
    let check_inputs = |param: &Param<'_>, arg| {
        match param.pat.kind {
            PatKind::Binding(_, id, ..) if path_to_local_id(arg, id) => {},
            _ => return false,
        }
        // checks that parameters are not bound as `ref` or `ref mut`
        if let Some(BindingMode::BindByReference(_)) = binding_modes.get(param.pat.hir_id) {
            return false;
        }

        match *cx.typeck_results().expr_adjustments(arg) {
            [] => true,
            [
                Adjustment {
                    kind: Adjust::Deref(None),
                    ..
                },
                Adjustment {
                    kind: Adjust::Borrow(AutoBorrow::Ref(_, mu2)),
                    ..
                },
            ] => {
                // re-borrow with the same mutability is allowed
                let ty = cx.typeck_results().expr_ty(arg);
                matches!(*ty.kind(), ty::Ref(.., mu1) if mu1 == mu2.into())
            },
            _ => false,
        }
    };
    std::iter::zip(params, receiver.into_iter().chain(call_args.iter())).all(|(param, arg)| check_inputs(param, arg))
}

fn check_sig<'tcx>(cx: &LateContext<'tcx>, closure_ty: Ty<'tcx>, call_ty: Ty<'tcx>) -> bool {
    let call_sig = call_ty.fn_sig(cx.tcx);
    if call_sig.unsafety() == Unsafety::Unsafe {
        return false;
    }
    if !closure_ty.has_late_bound_regions() {
        return true;
    }
    let ty::Closure(_, args) = closure_ty.kind() else {
        return false;
    };
    let closure_sig = cx.tcx.signature_unclosure(args.as_closure().sig(), Unsafety::Normal);
    cx.tcx.erase_late_bound_regions(closure_sig) == cx.tcx.erase_late_bound_regions(call_sig)
}

fn get_ufcs_type_name<'tcx>(cx: &LateContext<'tcx>, method_def_id: DefId, args: GenericArgsRef<'tcx>) -> String {
    let assoc_item = cx.tcx.associated_item(method_def_id);
    let def_id = assoc_item.container_id(cx.tcx);
    match assoc_item.container {
        ty::TraitContainer => cx.tcx.def_path_str(def_id),
        ty::ImplContainer => {
            let ty = cx.tcx.type_of(def_id).skip_binder();
            match ty.kind() {
                ty::Adt(adt, _) => cx.tcx.def_path_str(adt.did()),
                ty::Array(..)
                | ty::Dynamic(..)
                | ty::Never
                | ty::RawPtr(_)
                | ty::Ref(..)
                | ty::Slice(_)
                | ty::Tuple(_) => {
                    format!("<{}>", EarlyBinder::bind(ty).instantiate(cx.tcx, args))
                },
                _ => ty.to_string(),
            }
        },
    }
}
