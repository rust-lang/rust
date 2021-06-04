use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::{ast_utils, is_direct_expn_of};
use rustc_ast::ast::{Expr, ExprKind, Lit, LitKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** This lint warns about boolean comparisons in assert-like macros.
    ///
    /// **Why is this bad?** It is shorter to use the equivalent.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// // Bad
    /// assert_eq!("a".is_empty(), false);
    /// assert_ne!("a".is_empty(), true);
    ///
    /// // Good
    /// assert!(!"a".is_empty());
    /// ```
    pub BOOL_ASSERT_COMPARISON,
    style,
    "Using a boolean as comparison value in an assert_* macro when there is no need"
}

declare_lint_pass!(BoolAssertComparison => [BOOL_ASSERT_COMPARISON]);

fn is_bool_lit(e: &Expr) -> bool {
    matches!(
        e.kind,
        ExprKind::Lit(Lit {
            kind: LitKind::Bool(_),
            ..
        })
    ) && !e.span.from_expansion()
}

impl EarlyLintPass for BoolAssertComparison {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &Expr) {
        let macros = ["assert_eq", "debug_assert_eq"];
        let inverted_macros = ["assert_ne", "debug_assert_ne"];

        for mac in macros.iter().chain(inverted_macros.iter()) {
            if let Some(span) = is_direct_expn_of(e.span, mac) {
                if let Some([a, b]) = ast_utils::extract_assert_macro_args(e) {
                    let nb_bool_args = is_bool_lit(a) as usize + is_bool_lit(b) as usize;

                    if nb_bool_args != 1 {
                        // If there are two boolean arguments, we definitely don't understand
                        // what's going on, so better leave things as is...
                        //
                        // Or there is simply no boolean and then we can leave things as is!
                        return;
                    }

                    let non_eq_mac = &mac[..mac.len() - 3];
                    span_lint_and_sugg(
                        cx,
                        BOOL_ASSERT_COMPARISON,
                        span,
                        &format!("used `{}!` with a literal bool", mac),
                        "replace it with",
                        format!("{}!(..)", non_eq_mac),
                        Applicability::MaybeIncorrect,
                    );
                    return;
                }
            }
        }
    }
}
