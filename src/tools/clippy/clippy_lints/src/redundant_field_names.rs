use crate::utils::span_lint_and_sugg;
use rustc_ast::ast::{Expr, ExprKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for fields in struct literals where shorthands
    /// could be used.
    ///
    /// **Why is this bad?** If the field and variable names are the same,
    /// the field name is redundant.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let bar: u8 = 123;
    ///
    /// struct Foo {
    ///     bar: u8,
    /// }
    ///
    /// let foo = Foo { bar: bar };
    /// ```
    /// the last line can be simplified to
    /// ```ignore
    /// let foo = Foo { bar };
    /// ```
    pub REDUNDANT_FIELD_NAMES,
    style,
    "checks for fields in struct literals where shorthands could be used"
}

declare_lint_pass!(RedundantFieldNames => [REDUNDANT_FIELD_NAMES]);

impl EarlyLintPass for RedundantFieldNames {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::Struct(_, ref fields, _) = expr.kind {
            for field in fields {
                if field.is_shorthand {
                    continue;
                }
                if let ExprKind::Path(None, path) = &field.expr.kind {
                    if path.segments.len() == 1
                        && path.segments[0].ident == field.ident
                        && path.segments[0].args.is_none()
                    {
                        span_lint_and_sugg(
                            cx,
                            REDUNDANT_FIELD_NAMES,
                            field.span,
                            "redundant field names in struct initialization",
                            "replace it with",
                            field.ident.to_string(),
                            Applicability::MachineApplicable,
                        );
                    }
                }
            }
        }
    }
}
