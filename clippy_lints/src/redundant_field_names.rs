use rustc::lint::*;
use rustc::{declare_lint, lint_array};
use rustc::hir::*;
use crate::utils::{in_macro, is_range_expression, match_var, span_lint_and_sugg};

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
/// let foo = Foo{ bar: bar }
/// ```
declare_clippy_lint! {
    pub REDUNDANT_FIELD_NAMES,
    style,
    "checks for fields in struct literals where shorthands could be used"
}

pub struct RedundantFieldNames;

impl LintPass for RedundantFieldNames {
    fn get_lints(&self) -> LintArray {
        lint_array!(REDUNDANT_FIELD_NAMES)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for RedundantFieldNames {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        // Ignore all macros including range expressions.
        // They can have redundant field names when expanded.
        // e.g. range expression `start..end` is desugared to `Range { start: start, end: end }`
        if in_macro(expr.span) || is_range_expression(expr.span) {
            return;
        }

        if let ExprKind::Struct(_, ref fields, _) = expr.node {
            for field in fields {
                let name = field.ident.name;

                if match_var(&field.expr, name) && !field.is_shorthand {
                    span_lint_and_sugg (
                        cx,
                        REDUNDANT_FIELD_NAMES,
                        field.span,
                        "redundant field names in struct initialization",
                        "replace it with",
                        name.to_string()
                    );
                }
            }
        }
    }
}
