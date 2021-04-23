use crate::consts::constant_simple;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{indent_of, reindent_multiline, snippet_opt};
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::usage::contains_return_break_continue_macro;
use clippy_utils::{in_constant, match_qpath, path_to_local_id, paths, sugg};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Arm, Expr, ExprKind, Pat, PatKind};
use rustc_lint::LintContext;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// **What it does:**
    /// Finds patterns that reimplement `Option::unwrap_or` or `Result::unwrap_or`.
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
    "finds patterns that can be encoded more concisely with `Option::unwrap_or` or `Result::unwrap_or`"
}

declare_lint_pass!(ManualUnwrapOr => [MANUAL_UNWRAP_OR]);

impl LateLintPass<'_> for ManualUnwrapOr {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if in_external_macro(cx.sess(), expr.span) || in_constant(cx, expr.hir_id) {
            return;
        }
        lint_manual_unwrap_or(cx, expr);
    }
}

#[derive(Copy, Clone)]
enum Case {
    Option,
    Result,
}

impl Case {
    fn unwrap_fn_path(&self) -> &str {
        match self {
            Case::Option => "Option::unwrap_or",
            Case::Result => "Result::unwrap_or",
        }
    }
}

fn lint_manual_unwrap_or<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
    fn applicable_or_arm<'a>(arms: &'a [Arm<'a>]) -> Option<&'a Arm<'a>> {
        if_chain! {
            if arms.len() == 2;
            if arms.iter().all(|arm| arm.guard.is_none());
            if let Some((idx, or_arm)) = arms.iter().enumerate().find(|(_, arm)|
                match arm.pat.kind {
                    PatKind::Path(ref some_qpath) =>
                        match_qpath(some_qpath, &paths::OPTION_NONE),
                    PatKind::TupleStruct(ref err_qpath, &[Pat { kind: PatKind::Wild, .. }], _) =>
                        match_qpath(err_qpath, &paths::RESULT_ERR),
                    _ => false,
                }
            );
            let unwrap_arm = &arms[1 - idx];
            if let PatKind::TupleStruct(ref unwrap_qpath, &[unwrap_pat], _) = unwrap_arm.pat.kind;
            if match_qpath(unwrap_qpath, &paths::OPTION_SOME)
                || match_qpath(unwrap_qpath, &paths::RESULT_OK);
            if let PatKind::Binding(_, binding_hir_id, ..) = unwrap_pat.kind;
            if path_to_local_id(unwrap_arm.body, binding_hir_id);
            if !contains_return_break_continue_macro(or_arm.body);
            then {
                Some(or_arm)
            } else {
                None
            }
        }
    }

    if_chain! {
        if let ExprKind::Match(scrutinee, match_arms, _) = expr.kind;
        let ty = cx.typeck_results().expr_ty(scrutinee);
        if let Some(case) = if is_type_diagnostic_item(cx, ty, sym::option_type) {
            Some(Case::Option)
        } else if is_type_diagnostic_item(cx, ty, sym::result_type) {
            Some(Case::Result)
        } else {
            None
        };
        if let Some(or_arm) = applicable_or_arm(match_arms);
        if let Some(or_body_snippet) = snippet_opt(cx, or_arm.body.span);
        if let Some(indent) = indent_of(cx, expr.span);
        if constant_simple(cx, cx.typeck_results(), or_arm.body).is_some();
        then {
            let reindented_or_body =
                reindent_multiline(or_body_snippet.into(), true, Some(indent));
            span_lint_and_sugg(
                cx,
                MANUAL_UNWRAP_OR, expr.span,
                &format!("this pattern reimplements `{}`", case.unwrap_fn_path()),
                "replace with",
                format!(
                    "{}.unwrap_or({})",
                    sugg::Sugg::hir(cx, scrutinee, "..").maybe_par(),
                    reindented_or_body,
                ),
                Applicability::MachineApplicable,
            );
        }
    }
}
