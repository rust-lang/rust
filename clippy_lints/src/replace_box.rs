use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::{MaybeDef, MaybeResPath};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::implements_trait;
use clippy_utils::{is_default_equivalent_call, local_is_initialized, path_to_local};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Detects assignments of `Default::default()` or `Box::new(value)`
    /// to a place of type `Box<T>`.
    ///
    /// ### Why is this bad?
    /// This incurs an extra heap allocation compared to assigning the boxed
    /// storage.
    ///
    /// ### Example
    /// ```no_run
    /// let mut b = Box::new(1u32);
    /// b = Default::default();
    /// ```
    /// Use instead:
    /// ```no_run
    /// let mut b = Box::new(1u32);
    /// *b = Default::default();
    /// ```
    #[clippy::version = "1.92.0"]
    pub REPLACE_BOX,
    perf,
    "assigning a newly created box to `Box<T>` is inefficient"
}
declare_lint_pass!(ReplaceBox => [REPLACE_BOX]);

impl LateLintPass<'_> for ReplaceBox {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'_ Expr<'_>) {
        if let ExprKind::Assign(lhs, rhs, _) = &expr.kind
            && !lhs.span.from_expansion()
            && !rhs.span.from_expansion()
            && let lhs_ty = cx.typeck_results().expr_ty(lhs)
            // No diagnostic for late-initialized locals
            && path_to_local(lhs).is_none_or(|local| local_is_initialized(cx, local))
            && let Some(inner_ty) = lhs_ty.boxed_ty()
        {
            if let Some(default_trait_id) = cx.tcx.get_diagnostic_item(sym::Default)
                && implements_trait(cx, inner_ty, default_trait_id, &[])
                && is_default_call(cx, rhs)
            {
                span_lint_and_then(
                    cx,
                    REPLACE_BOX,
                    expr.span,
                    "creating a new box with default content",
                    |diag| {
                        let mut app = Applicability::MachineApplicable;
                        let suggestion = format!(
                            "{} = Default::default()",
                            Sugg::hir_with_applicability(cx, lhs, "_", &mut app).deref()
                        );

                        diag.note("this creates a needless allocation").span_suggestion(
                            expr.span,
                            "replace existing content with default instead",
                            suggestion,
                            app,
                        );
                    },
                );
            }

            if inner_ty.is_sized(cx.tcx, cx.typing_env())
                && let Some(rhs_inner) = get_box_new_payload(cx, rhs)
            {
                span_lint_and_then(cx, REPLACE_BOX, expr.span, "creating a new box", |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let suggestion = format!(
                        "{} = {}",
                        Sugg::hir_with_applicability(cx, lhs, "_", &mut app).deref(),
                        Sugg::hir_with_context(cx, rhs_inner, expr.span.ctxt(), "_", &mut app),
                    );

                    diag.note("this creates a needless allocation").span_suggestion(
                        expr.span,
                        "replace existing content with inner value instead",
                        suggestion,
                        app,
                    );
                });
            }
        }
    }
}

fn is_default_call(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    matches!(expr.kind, ExprKind::Call(func, _args) if is_default_equivalent_call(cx, func, Some(expr)))
}

fn get_box_new_payload<'tcx>(cx: &LateContext<'_>, expr: &Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    if let ExprKind::Call(box_new, [arg]) = expr.kind
        && let ExprKind::Path(QPath::TypeRelative(ty, seg)) = box_new.kind
        && seg.ident.name == sym::new
        && ty.basic_res().is_lang_item(cx, LangItem::OwnedBox)
    {
        Some(arg)
    } else {
        None
    }
}
