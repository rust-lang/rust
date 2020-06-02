use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Block, ExprKind, PatKind, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

use crate::utils::{in_macro, match_qpath, snippet_opt, span_lint_and_then};

declare_clippy_lint! {
    /// **What it does:** Checks for `let`-bindings, which are subsequently
    /// returned.
    ///
    /// **Why is this bad?** It is just extraneous code. Remove it to make your code
    /// more rusty.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// fn foo() -> String {
    ///     let x = String::new();
    ///     x
    /// }
    /// ```
    /// instead, use
    /// ```
    /// fn foo() -> String {
    ///     String::new()
    /// }
    /// ```
    pub LET_AND_RETURN,
    style,
    "creating a let-binding and then immediately returning it like `let x = expr; x` at the end of a block"
}

declare_lint_pass!(LetReturn => [LET_AND_RETURN]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for LetReturn {
    fn check_block(&mut self, cx: &LateContext<'a, 'tcx>, block: &'tcx Block<'_>) {
        // we need both a let-binding stmt and an expr
        if_chain! {
            if let Some(retexpr) = block.expr;
            if let Some(stmt) = block.stmts.iter().last();
            if let StmtKind::Local(local) = &stmt.kind;
            if local.ty.is_none();
            if local.attrs.is_empty();
            if let Some(initexpr) = &local.init;
            if let PatKind::Binding(.., ident, _) = local.pat.kind;
            if let ExprKind::Path(qpath) = &retexpr.kind;
            if match_qpath(qpath, &[&*ident.name.as_str()]);
            if !in_external_macro(cx.sess(), initexpr.span);
            if !in_external_macro(cx.sess(), retexpr.span);
            if !in_external_macro(cx.sess(), local.span);
            if !in_macro(local.span);
            then {
                span_lint_and_then(
                    cx,
                    LET_AND_RETURN,
                    retexpr.span,
                    "returning the result of a `let` binding from a block",
                    |err| {
                        err.span_label(local.span, "unnecessary `let` binding");

                        if let Some(snippet) = snippet_opt(cx, initexpr.span) {
                            err.multipart_suggestion(
                                "return the expression directly",
                                vec![
                                    (local.span, String::new()),
                                    (retexpr.span, snippet),
                                ],
                                Applicability::MachineApplicable,
                            );
                        } else {
                            err.span_help(initexpr.span, "this expression can be directly returned");
                        }
                    },
                );
            }
        }
    }
}
