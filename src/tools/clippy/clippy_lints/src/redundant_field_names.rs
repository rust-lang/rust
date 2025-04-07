use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, MsrvStack};
use rustc_ast::ast::{Expr, ExprKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::impl_lint_pass;

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
    /// ```no_run
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
    msrv: MsrvStack,
}

impl RedundantFieldNames {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            msrv: MsrvStack::new(conf.msrv),
        }
    }
}

impl_lint_pass!(RedundantFieldNames => [REDUNDANT_FIELD_NAMES]);

impl EarlyLintPass for RedundantFieldNames {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if !self.msrv.meets(msrvs::FIELD_INIT_SHORTHAND) {
            return;
        }

        if expr.span.in_external_macro(cx.sess().source_map()) {
            return;
        }
        if let ExprKind::Struct(ref se) = expr.kind {
            for field in &se.fields {
                if !field.is_shorthand
                    && let ExprKind::Path(None, path) = &field.expr.kind
                    && let [segment] = path.segments.as_slice()
                    && segment.args.is_none()
                    && segment.ident == field.ident
                    && field.span.eq_ctxt(field.ident.span)
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

    extract_msrv_attr!();
}
