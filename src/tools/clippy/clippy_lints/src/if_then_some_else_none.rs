use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::source::snippet_with_macro_callsite;
use clippy_utils::{is_else_clause, is_lang_ctor, meets_msrv, msrvs};
use if_chain::if_chain;
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// **What it does:** Checks for if-else that could be written to `bool::then`.
    ///
    /// **Why is this bad?** Looks a little redundant. Using `bool::then` helps it have less lines of code.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// # let v = vec![0];
    /// let a = if v.is_empty() {
    ///     println!("true!");
    ///     Some(42)
    /// } else {
    ///     None
    /// };
    /// ```
    ///
    /// Could be written:
    ///
    /// ```rust
    /// # let v = vec![0];
    /// let a = v.is_empty().then(|| {
    ///     println!("true!");
    ///     42
    /// });
    /// ```
    pub IF_THEN_SOME_ELSE_NONE,
    restriction,
    "Finds if-else that could be written using `bool::then`"
}

pub struct IfThenSomeElseNone {
    msrv: Option<RustcVersion>,
}

impl IfThenSomeElseNone {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(IfThenSomeElseNone => [IF_THEN_SOME_ELSE_NONE]);

impl LateLintPass<'_> for IfThenSomeElseNone {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'tcx Expr<'_>) {
        if !meets_msrv(self.msrv.as_ref(), &msrvs::BOOL_THEN) {
            return;
        }

        if in_external_macro(cx.sess(), expr.span) {
            return;
        }

        // We only care about the top-most `if` in the chain
        if is_else_clause(cx.tcx, expr) {
            return;
        }

        if_chain! {
            if let ExprKind::If(cond, then, Some(els)) = expr.kind;
            if let ExprKind::Block(then_block, _) = then.kind;
            if let Some(then_expr) = then_block.expr;
            if let ExprKind::Call(then_call, [then_arg]) = then_expr.kind;
            if let ExprKind::Path(ref then_call_qpath) = then_call.kind;
            if is_lang_ctor(cx, then_call_qpath, OptionSome);
            if let ExprKind::Block(els_block, _) = els.kind;
            if els_block.stmts.is_empty();
            if let Some(els_expr) = els_block.expr;
            if let ExprKind::Path(ref qpath) = els_expr.kind;
            if is_lang_ctor(cx, qpath, OptionNone);
            then {
                let cond_snip = snippet_with_macro_callsite(cx, cond.span, "[condition]");
                let cond_snip = if matches!(cond.kind, ExprKind::Unary(_, _) | ExprKind::Binary(_, _, _)) {
                    format!("({})", cond_snip)
                } else {
                    cond_snip.into_owned()
                };
                let arg_snip = snippet_with_macro_callsite(cx, then_arg.span, "");
                let closure_body = if then_block.stmts.is_empty() {
                    arg_snip.into_owned()
                } else {
                    format!("{{ /* snippet */ {} }}", arg_snip)
                };
                let help = format!(
                    "consider using `bool::then` like: `{}.then(|| {})`",
                    cond_snip,
                    closure_body,
                );
                span_lint_and_help(
                    cx,
                    IF_THEN_SOME_ELSE_NONE,
                    expr.span,
                    "this could be simplified with `bool::then`",
                    None,
                    &help,
                );
            }
        }
    }

    extract_msrv_attr!(LateContext);
}
