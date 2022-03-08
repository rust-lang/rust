use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::{meets_msrv, msrvs};
use rustc_ast::ast::{Expr, ExprKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for fields in struct literals where shorthands
    /// could be used.
    ///
    /// ### Why is this bad?
    /// If the field and variable names are the same,
    /// the field name is redundant.
    ///
    /// ### Example
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
    #[clippy::version = "pre 1.29.0"]
    pub REDUNDANT_FIELD_NAMES,
    style,
    "checks for fields in struct literals where shorthands could be used"
}

pub struct RedundantFieldNames {
    msrv: Option<RustcVersion>,
}

impl RedundantFieldNames {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(RedundantFieldNames => [REDUNDANT_FIELD_NAMES]);

impl EarlyLintPass for RedundantFieldNames {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if !meets_msrv(self.msrv.as_ref(), &msrvs::FIELD_INIT_SHORTHAND) {
            return;
        }

        if in_external_macro(cx.sess(), expr.span) {
            return;
        }
        if let ExprKind::Struct(ref se) = expr.kind {
            for field in &se.fields {
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
    extract_msrv_attr!(EarlyContext);
}
