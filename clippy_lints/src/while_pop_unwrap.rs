use clippy_utils::{diagnostics::span_lint_and_then, match_def_path, paths, source::snippet};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, PatKind, Stmt, StmtKind, UnOp};
use rustc_lint::LateContext;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{symbol::Ident, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Looks for loops that check for emptiness of a `Vec` in the condition and pop an element
    /// in the body as a separate operation.
    ///
    /// ### Why is this bad?
    /// Such loops can be written in a more idiomatic way by using a while..let loop and directly
    /// pattern matching on the return value of `Vec::pop()`.
    ///
    /// ### Example
    /// ```rust
    /// let mut numbers = vec![1, 2, 3, 4, 5];
    /// while !numbers.is_empty() {
    ///     let number = numbers.pop().unwrap();
    ///     // use `number`
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// let mut numbers = vec![1, 2, 3, 4, 5];
    /// while let Some(number) = numbers.pop() {
    ///     // use `number`
    /// }
    /// ```
    #[clippy::version = "1.70.0"]
    pub WHILE_POP_UNWRAP,
    style,
    "checking for emptiness of a `Vec` in the loop condition and popping an element in the body"
}
declare_lint_pass!(WhilePopUnwrap => [WHILE_POP_UNWRAP]);

fn report_lint(cx: &LateContext<'_>, pop_span: Span, ident: Option<Ident>, loop_span: Span, receiver_span: Span) {
    span_lint_and_then(
        cx,
        WHILE_POP_UNWRAP,
        pop_span,
        "you seem to be trying to pop elements from a `Vec` in a loop",
        |diag| {
            diag.span_suggestion(
                loop_span,
                "try",
                format!(
                    "while let Some({}) = {}.pop()",
                    ident.as_ref().map_or("element", Ident::as_str),
                    snippet(cx, receiver_span, "..")
                ),
                Applicability::MaybeIncorrect,
            )
            .note("this while loop can be written in a more idiomatic way");
        },
    );
}

fn match_method_call(cx: &LateContext<'_>, expr: &Expr<'_>, method: &[&str]) -> bool {
    if let ExprKind::MethodCall(..) = expr.kind
        && let Some(id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && match_def_path(cx, id, method)
    {
        true
    } else {
        false
    }
}

fn is_vec_pop(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    match_method_call(cx, expr, &paths::VEC_POP)
}

fn is_vec_pop_unwrap(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let ExprKind::MethodCall(_, inner, ..) = expr.kind
        && (match_method_call(cx, expr, &paths::OPTION_UNWRAP) || match_method_call(cx, expr, &paths::OPTION_EXPECT))
        && is_vec_pop(cx, inner)
    {
        true
    } else {
        false
    }
}

fn check_local(cx: &LateContext<'_>, stmt: &Stmt<'_>, loop_span: Span, recv_span: Span) {
    if let StmtKind::Local(local) = stmt.kind
        && let PatKind::Binding(.., ident, _) = local.pat.kind
        && let Some(init) = local.init
        && let ExprKind::MethodCall(_, inner, ..) = init.kind
        && is_vec_pop_unwrap(cx, init)
    {
        report_lint(cx, init.span.to(inner.span), Some(ident), loop_span, recv_span);
    }
}

fn check_call_arguments(cx: &LateContext<'_>, stmt: &Stmt<'_>, loop_span: Span, recv_span: Span) {
    if let StmtKind::Semi(expr) | StmtKind::Expr(expr) = stmt.kind {
        if let ExprKind::MethodCall(_, _, args, _) | ExprKind::Call(_, args) = expr.kind {
            let offending_arg = args
                .iter()
                .find_map(|arg| is_vec_pop_unwrap(cx, arg).then_some(arg.span));

            if let Some(offending_arg) = offending_arg {
                report_lint(cx, offending_arg, None, loop_span, recv_span);
            }
        }
    }
}

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, cond: &'tcx Expr<'_>, body: &'tcx Expr<'_>) {
    if let ExprKind::Unary(UnOp::Not, cond) = cond.kind
        && let ExprKind::MethodCall(_, Expr { span: recv_span, .. }, _, _) = cond.kind
        && match_method_call(cx, cond, &paths::VEC_IS_EMPTY)
        && let ExprKind::Block(body, _) = body.kind
        && let Some(stmt) = body.stmts.first()
    {
        check_local(cx, stmt, cond.span, *recv_span);
        check_call_arguments(cx, stmt, cond.span, *recv_span);
    }
}
