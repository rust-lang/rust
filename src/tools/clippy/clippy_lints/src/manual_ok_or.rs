use crate::utils::{
    indent_of, is_type_diagnostic_item, match_qpath, paths, reindent_multiline, snippet_opt, span_lint_and_sugg,
};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{def, Expr, ExprKind, PatKind, QPath};
use rustc_lint::LintContext;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:**
    /// Finds patterns that reimplement `Option::ok_or`.
    ///
    /// **Why is this bad?**
    /// Concise code helps focusing on behavior instead of boilerplate.
    ///
    /// **Known problems:** None.
    ///
    /// **Examples:**
    /// ```rust
    /// let foo: Option<i32> = None;
    /// foo.map_or(Err("error"), |v| Ok(v));
    ///
    /// let foo: Option<i32> = None;
    /// foo.map_or(Err("error"), |v| Ok(v));
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let foo: Option<i32> = None;
    /// foo.ok_or("error");
    /// ```
    pub MANUAL_OK_OR,
    pedantic,
    "finds patterns that can be encoded more concisely with `Option::ok_or`"
}

declare_lint_pass!(ManualOkOr => [MANUAL_OK_OR]);

impl LateLintPass<'_> for ManualOkOr {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, scrutinee: &'tcx Expr<'tcx>) {
        if in_external_macro(cx.sess(), scrutinee.span) {
            return;
        }

        if_chain! {
            if let ExprKind::MethodCall(method_segment, _, args, _) = scrutinee.kind;
            if method_segment.ident.name == sym!(map_or);
            if args.len() == 3;
            let method_receiver = &args[0];
            let ty = cx.typeck_results().expr_ty(method_receiver);
            if is_type_diagnostic_item(cx, ty, sym!(option_type));
            let or_expr = &args[1];
            if is_ok_wrapping(cx, &args[2]);
            if let ExprKind::Call(Expr { kind: ExprKind::Path(err_path), .. }, &[ref err_arg]) = or_expr.kind;
            if match_qpath(err_path, &paths::RESULT_ERR);
            if let Some(method_receiver_snippet) = snippet_opt(cx, method_receiver.span);
            if let Some(err_arg_snippet) = snippet_opt(cx, err_arg.span);
            if let Some(indent) = indent_of(cx, scrutinee.span);
            then {
                let reindented_err_arg_snippet =
                    reindent_multiline(err_arg_snippet.into(), true, Some(indent + 4));
                span_lint_and_sugg(
                    cx,
                    MANUAL_OK_OR,
                    scrutinee.span,
                    "this pattern reimplements `Option::ok_or`",
                    "replace with",
                    format!(
                        "{}.ok_or({})",
                        method_receiver_snippet,
                        reindented_err_arg_snippet
                    ),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}

fn is_ok_wrapping(cx: &LateContext<'_>, map_expr: &Expr<'_>) -> bool {
    if let ExprKind::Path(ref qpath) = map_expr.kind {
        if match_qpath(qpath, &paths::RESULT_OK) {
            return true;
        }
    }
    if_chain! {
        if let ExprKind::Closure(_, _, body_id, ..) = map_expr.kind;
        let body = cx.tcx.hir().body(body_id);
        if let PatKind::Binding(_, param_id, ..) = body.params[0].pat.kind;
        if let ExprKind::Call(Expr { kind: ExprKind::Path(ok_path), .. }, &[ref ok_arg]) = body.value.kind;
        if match_qpath(ok_path, &paths::RESULT_OK);
        if let ExprKind::Path(QPath::Resolved(_, ok_arg_path)) = ok_arg.kind;
        if let def::Res::Local(ok_arg_path_id) = ok_arg_path.res;
        then { param_id == ok_arg_path_id } else { false }
    }
}
