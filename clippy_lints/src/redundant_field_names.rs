use rustc::lint::*;
use rustc::hir::*;
use utils::{span_lint_and_sugg};

/// **What it does:** Checks for redundnat field names where shorthands
/// can be used.
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
declare_lint! {
    pub REDUNDANT_FIELD_NAMES,
    Warn,
    "using same name for field and variable ,where shorthand can be used"
}

pub struct RedundantFieldNames;

impl LintPass for RedundantFieldNames {
    fn get_lints(&self) -> LintArray {
        lint_array!(REDUNDANT_FIELD_NAMES)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for RedundantFieldNames {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprStruct(_, ref fields, _) = expr.node {
            for field in fields {
                let name = field.name.node;
                if let ExprPath(ref qpath) = field.expr.node {
                    if let &QPath::Resolved(_, ref path) = qpath {
                        let segments = &path.segments;

                        if segments.len() == 1 {
                            let expr_name = segments[0].name;

                            if name == expr_name {
                                span_lint_and_sugg(
                                    cx,
                                    REDUNDANT_FIELD_NAMES,
                                    path.span,
                                    "redundant field names in struct initialization",
                                    &format!(
                                        "replace '{0}: {0}' with '{0}'",
                                        name,
                                    ),
                                    "".to_string()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
