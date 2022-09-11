use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::macros::{find_assert_args, root_macro_call_first_node, PanicExpn};
use clippy_utils::path_res;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::{implements_trait, is_copy, is_type_diagnostic_item};
use clippy_utils::usage::local_used_after_expr;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `assert!(r.is_ok())` or `assert!(r.is_err())` calls.
    ///
    /// ### Why is this bad?
    /// An assertion failure cannot output an useful message of the error.
    ///
    /// ### Known problems
    /// The suggested replacement decreases the readability of code and log output.
    ///
    /// ### Example
    /// ```rust,ignore
    /// # let r = Ok::<_, ()>(());
    /// assert!(r.is_ok());
    /// # let r = Err::<_, ()>(());
    /// assert!(r.is_err());
    /// ```
    #[clippy::version = "1.64.0"]
    pub ASSERTIONS_ON_RESULT_STATES,
    restriction,
    "`assert!(r.is_ok())`/`assert!(r.is_err())` gives worse error message than directly calling `r.unwrap()`/`r.unwrap_err()`"
}

declare_lint_pass!(AssertionsOnResultStates => [ASSERTIONS_ON_RESULT_STATES]);

impl<'tcx> LateLintPass<'tcx> for AssertionsOnResultStates {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let Some(macro_call) = root_macro_call_first_node(cx, e)
            && matches!(cx.tcx.get_diagnostic_name(macro_call.def_id), Some(sym::assert_macro))
            && let Some((condition, panic_expn)) = find_assert_args(cx, e, macro_call.expn)
            && matches!(panic_expn, PanicExpn::Empty)
            && let ExprKind::MethodCall(method_segment, recv, [], _) = condition.kind
            && let result_type_with_refs = cx.typeck_results().expr_ty(recv)
            && let result_type = result_type_with_refs.peel_refs()
            && is_type_diagnostic_item(cx, result_type, sym::Result)
            && let ty::Adt(_, substs) = result_type.kind()
        {
            if !is_copy(cx, result_type) {
                if result_type_with_refs != result_type {
                    return;
                } else if let Res::Local(binding_id) = path_res(cx, recv)
                    && local_used_after_expr(cx, binding_id, recv)
                {
                    return;
                }
            }
            let mut app = Applicability::MachineApplicable;
            match method_segment.ident.as_str() {
                "is_ok" if type_suitable_to_unwrap(cx, substs.type_at(1)) => {
                    span_lint_and_sugg(
                        cx,
                        ASSERTIONS_ON_RESULT_STATES,
                        macro_call.span,
                        "called `assert!` with `Result::is_ok`",
                        "replace with",
                        format!(
                            "{}.unwrap()",
                            snippet_with_context(cx, recv.span, condition.span.ctxt(), "..", &mut app).0
                        ),
                        app,
                    );
                }
                "is_err" if type_suitable_to_unwrap(cx, substs.type_at(0)) => {
                    span_lint_and_sugg(
                        cx,
                        ASSERTIONS_ON_RESULT_STATES,
                        macro_call.span,
                        "called `assert!` with `Result::is_err`",
                        "replace with",
                        format!(
                            "{}.unwrap_err()",
                            snippet_with_context(cx, recv.span, condition.span.ctxt(), "..", &mut app).0
                        ),
                        app,
                    );
                }
                _ => (),
            };
        }
    }
}

/// This checks whether a given type is known to implement Debug.
fn has_debug_impl<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
    cx.tcx
        .get_diagnostic_item(sym::Debug)
        .map_or(false, |debug| implements_trait(cx, ty, debug, &[]))
}

fn type_suitable_to_unwrap<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
    has_debug_impl(cx, ty) && !ty.is_unit() && !ty.is_never()
}
