use crate::{EarlyContext, EarlyLintPass, LintContext};
use rustc_ast as ast;
use rustc_errors::Applicability;
use rustc_span::source_map::ExpnKind;

declare_lint! {
    /// The `redundant_field_initializers` lint checks for fields in struct literals
    /// where shorthands could be used.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let bar: u8 = 123;
    ///
    /// struct Foo {
    ///     bar: u8,
    /// }
    ///
    /// let foo = Foo { bar: bar };
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// If the field and variable names are the same,
    /// the field name is redundant.
    pub REDUNDANT_FIELD_INITIALIZERS,
    Warn,
    "checks for fields in struct literals where shorthands could be used"
}

declare_lint_pass!(RedundantFieldInitializers => [REDUNDANT_FIELD_INITIALIZERS]);

impl EarlyLintPass for RedundantFieldInitializers {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &ast::Expr) {
        if let ExpnKind::Macro(..) = expr.span.ctxt().outer_expn_data().kind {
            // Do not lint on macro output.
            return;
        }

        if let ast::ExprKind::Struct(ref se) = expr.kind {
            for field in &se.fields {
                if field.is_shorthand {
                    continue;
                }
                if let ast::ExprKind::Path(None, path) = &field.expr.kind {
                    if path.segments.len() == 1
                        && path.segments[0].ident == field.ident
                        && path.segments[0].args.is_none()
                    {
                        cx.struct_span_lint(REDUNDANT_FIELD_INITIALIZERS, field.span, |lint| {
                            lint.build("redundant field names in struct initialization")
                                .span_suggestion(
                                    field.span,
                                    "replace it with",
                                    field.ident.to_string(),
                                    Applicability::MachineApplicable,
                                )
                                .emit();
                        });
                    }
                }
            }
        }
    }
}
