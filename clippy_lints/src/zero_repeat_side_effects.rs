use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::VecArgs;
use clippy_utils::source::snippet;
use rustc_ast::LitKind;
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir::{ConstArgKind, ExprKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::IsSuggestable;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for array or vec initializations which contain an expression with side effects,
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
    if inner_expr.can_have_side_effects() {
        let parent_hir_node = cx.tcx.parent_hir_node(expr.hir_id);
        let inner_expr_ty = cx.typeck_results().expr_ty(inner_expr);
        let return_type = cx.typeck_results().expr_ty(expr);

        let inner_expr = snippet(cx, inner_expr.span.source_callsite(), "..");
        let vec = if is_vec { "vec!" } else { "" };

        let (span, sugg) = match parent_hir_node {
            Node::LetStmt(l) => (
                l.span,
                format!(
                    "{inner_expr}; let {var_name}: {return_type} = {vec}[];",
                    var_name = snippet(cx, l.pat.span.source_callsite(), "..")
                ),
            ),
            Node::Expr(x) if let ExprKind::Assign(l, _, _) = x.kind => (
                x.span,
                format!(
                    "{inner_expr}; {var_name} = {vec}[] as {return_type}",
                    var_name = snippet(cx, l.span.source_callsite(), "..")
                ),
            ),
            _ => (expr.span, format!("{{ {inner_expr}; {vec}[] as {return_type} }}")),
        };
        let span = span.source_callsite();
        span_lint_and_then(
            cx,
            ZERO_REPEAT_SIDE_EFFECTS,
            span,
            "expression with side effects as the initial value in a zero-sized array initializer",
            |diag| {
                if (!inner_expr_ty.is_never() || cx.tcx.features().never_type())
                    && return_type.is_suggestable(cx.tcx, true)
                {
                    diag.span_suggestion_verbose(
                        span,
                        "consider performing the side effect separately",
                        sugg,
                        Applicability::Unspecified,
                    );
                } else {
                    diag.help("consider performing the side effect separately");
                }
            },
        );
    }
}
