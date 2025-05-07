use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher::VecArgs;
use clippy_utils::source::snippet;
use clippy_utils::visitors::for_each_expr_without_closures;
use rustc_ast::LitKind;
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir::{ConstArgKind, ExprKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for array or vec initializations which call a function or method,
    /// but which have a repeat count of zero.
    ///
    /// ### Why is this bad?
    /// Such an initialization, despite having a repeat length of 0, will still call the inner function.
    /// This may not be obvious and as such there may be unintended side effects in code.
    ///
    /// ### Example
    /// ```no_run
    /// fn side_effect() -> i32 {
    ///     println!("side effect");
    ///     10
    /// }
    /// let a = [side_effect(); 0];
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn side_effect() -> i32 {
    ///     println!("side effect");
    ///     10
    /// }
    /// side_effect();
    /// let a: [i32; 0] = [];
    /// ```
    #[clippy::version = "1.79.0"]
    pub ZERO_REPEAT_SIDE_EFFECTS,
    suspicious,
    "usage of zero-sized initializations of arrays or vecs causing side effects"
}

declare_lint_pass!(ZeroRepeatSideEffects => [ZERO_REPEAT_SIDE_EFFECTS]);

impl LateLintPass<'_> for ZeroRepeatSideEffects {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &rustc_hir::Expr<'_>) {
        if let Some(args) = VecArgs::hir(cx, expr)
            && let VecArgs::Repeat(inner_expr, len) = args
            && let ExprKind::Lit(l) = len.kind
            && let LitKind::Int(Pu128(0), _) = l.node
        {
            inner_check(cx, expr, inner_expr, true);
        }
        // Lint only if the length is a literal zero, and not a path to any constants.
        // NOTE(@y21): When reading `[f(); LEN]`, I intuitively expect that the function is called and it
        // doesn't seem as confusing as `[f(); 0]`. It would also have false positives when eg.
        // the const item depends on `#[cfg]s` and has different values in different compilation
        // sessions).
        else if let ExprKind::Repeat(inner_expr, const_arg) = expr.kind
            && let ConstArgKind::Anon(anon_const) = const_arg.kind
            && let length_expr = cx.tcx.hir_body(anon_const.body).value
            && !length_expr.span.from_expansion()
            && let ExprKind::Lit(literal) = length_expr.kind
            && let LitKind::Int(Pu128(0), _) = literal.node
        {
            inner_check(cx, expr, inner_expr, false);
        }
    }
}

fn inner_check(cx: &LateContext<'_>, expr: &'_ rustc_hir::Expr<'_>, inner_expr: &'_ rustc_hir::Expr<'_>, is_vec: bool) {
    // check if expr is a call or has a call inside it
    if for_each_expr_without_closures(inner_expr, |x| {
        if let ExprKind::Call(_, _) | ExprKind::MethodCall(_, _, _, _) = x.kind {
            std::ops::ControlFlow::Break(())
        } else {
            std::ops::ControlFlow::Continue(())
        }
    })
    .is_some()
    {
        let parent_hir_node = cx.tcx.parent_hir_node(expr.hir_id);
        let return_type = cx.typeck_results().expr_ty(expr);

        if let Node::LetStmt(l) = parent_hir_node {
            array_span_lint(
                cx,
                l.span,
                inner_expr.span,
                l.pat.span,
                Some(return_type),
                is_vec,
                false,
            );
        } else if let Node::Expr(x) = parent_hir_node
            && let ExprKind::Assign(l, _, _) = x.kind
        {
            array_span_lint(cx, x.span, inner_expr.span, l.span, Some(return_type), is_vec, true);
        } else {
            span_lint_and_sugg(
                cx,
                ZERO_REPEAT_SIDE_EFFECTS,
                expr.span.source_callsite(),
                "function or method calls as the initial value in zero-sized array initializers may cause side effects",
                "consider using",
                format!(
                    "{{ {}; {}[] as {return_type} }}",
                    snippet(cx, inner_expr.span.source_callsite(), ".."),
                    if is_vec { "vec!" } else { "" },
                ),
                Applicability::Unspecified,
            );
        }
    }
}

fn array_span_lint(
    cx: &LateContext<'_>,
    expr_span: Span,
    func_call_span: Span,
    variable_name_span: Span,
    expr_ty: Option<Ty<'_>>,
    is_vec: bool,
    is_assign: bool,
) {
    let has_ty = expr_ty.is_some();

    span_lint_and_sugg(
        cx,
        ZERO_REPEAT_SIDE_EFFECTS,
        expr_span.source_callsite(),
        "function or method calls as the initial value in zero-sized array initializers may cause side effects",
        "consider using",
        format!(
            "{}; {}{}{} = {}[]{}{}",
            snippet(cx, func_call_span.source_callsite(), ".."),
            if has_ty && !is_assign { "let " } else { "" },
            snippet(cx, variable_name_span.source_callsite(), ".."),
            if let Some(ty) = expr_ty
                && !is_assign
            {
                format!(": {ty}")
            } else {
                String::new()
            },
            if is_vec { "vec!" } else { "" },
            if let Some(ty) = expr_ty
                && is_assign
            {
                format!(" as {ty}")
            } else {
                String::new()
            },
            if is_assign { "" } else { ";" }
        ),
        Applicability::Unspecified,
    );
}
