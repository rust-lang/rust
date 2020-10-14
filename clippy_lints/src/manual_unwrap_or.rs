use crate::consts::constant_simple;
use crate::utils;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{def, Arm, Expr, ExprKind, PatKind, QPath};
use rustc_lint::LintContext;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:**
    /// Finds patterns that reimplement `Option::unwrap_or`.
    ///
    /// **Why is this bad?**
    /// Concise code helps focusing on behavior instead of boilerplate.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let foo: Option<i32> = None;
    /// match foo {
    ///     Some(v) => v,
    ///     None => 1,
    /// };
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let foo: Option<i32> = None;
    /// foo.unwrap_or(1);
    /// ```
    pub MANUAL_UNWRAP_OR,
    complexity,
    "finds patterns that can be encoded more concisely with `Option::unwrap_or`"
}

declare_lint_pass!(ManualUnwrapOr => [MANUAL_UNWRAP_OR]);

impl LateLintPass<'_> for ManualUnwrapOr {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }
        lint_option_unwrap_or_case(cx, expr);
    }
}

fn lint_option_unwrap_or_case<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
    fn applicable_none_arm<'a>(arms: &'a [Arm<'a>]) -> Option<&'a Arm<'a>> {
        if_chain! {
            if arms.len() == 2;
            if arms.iter().all(|arm| arm.guard.is_none());
            if let Some((idx, none_arm)) = arms.iter().enumerate().find(|(_, arm)|
                if let PatKind::Path(ref qpath) = arm.pat.kind {
                    utils::match_qpath(qpath, &utils::paths::OPTION_NONE)
                } else {
                    false
                }
            );
            let some_arm = &arms[1 - idx];
            if let PatKind::TupleStruct(ref some_qpath, &[some_binding], _) = some_arm.pat.kind;
            if utils::match_qpath(some_qpath, &utils::paths::OPTION_SOME);
            if let PatKind::Binding(_, binding_hir_id, ..) = some_binding.kind;
            if let ExprKind::Path(QPath::Resolved(_, body_path)) = some_arm.body.kind;
            if let def::Res::Local(body_path_hir_id) = body_path.res;
            if body_path_hir_id == binding_hir_id;
            if !utils::usage::contains_return_break_continue_macro(none_arm.body);
            then {
                Some(none_arm)
            } else {
                None
            }
        }
    }

    if_chain! {
        if let ExprKind::Match(scrutinee, match_arms, _) = expr.kind;
        let ty = cx.typeck_results().expr_ty(scrutinee);
        if utils::is_type_diagnostic_item(cx, ty, sym!(option_type));
        if let Some(none_arm) = applicable_none_arm(match_arms);
        if let Some(scrutinee_snippet) = utils::snippet_opt(cx, scrutinee.span);
        if let Some(none_body_snippet) = utils::snippet_opt(cx, none_arm.body.span);
        if let Some(indent) = utils::indent_of(cx, expr.span);
        if constant_simple(cx, cx.typeck_results(), none_arm.body).is_some();
        then {
            let reindented_none_body =
                utils::reindent_multiline(none_body_snippet.into(), true, Some(indent));
            utils::span_lint_and_sugg(
                cx,
                MANUAL_UNWRAP_OR, expr.span,
                "this pattern reimplements `Option::unwrap_or`",
                "replace with",
                format!(
                    "{}.unwrap_or({})",
                    scrutinee_snippet,
                    reindented_none_body,
                ),
                Applicability::MachineApplicable,
            );
        }
    }
}
