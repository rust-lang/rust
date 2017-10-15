use consts::{constant_simple, Constant};
use rustc::hir::*;
use rustc::lint::*;
use syntax::codemap::Span;
use utils::{in_macro, span_lint};

/// **What it does:** Checks for erasing operations, e.g. `x * 0`.
///
/// **Why is this bad?** The whole expression can be replaced by zero.
/// This is most likely not the intended outcome and should probably be
/// corrected
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// 0 / x; 0 * x; x & 0
/// ```
declare_lint! {
    pub ERASING_OP,
    Warn,
    "using erasing operations, e.g. `x * 0` or `y & 0`"
}

#[derive(Copy, Clone)]
pub struct ErasingOp;

impl LintPass for ErasingOp {
    fn get_lints(&self) -> LintArray {
        lint_array!(ERASING_OP)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ErasingOp {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if in_macro(e.span) {
            return;
        }
        if let ExprBinary(ref cmp, ref left, ref right) = e.node {
            match cmp.node {
                BiMul | BiBitAnd => {
                    check(cx, left, e.span);
                    check(cx, right, e.span);
                },
                BiDiv => check(cx, left, e.span),
                _ => (),
            }
        }
    }
}

fn check(cx: &LateContext, e: &Expr, span: Span) {
    if let Some(Constant::Int(v)) = constant_simple(cx, e) {
        if v.to_u128_unchecked() == 0 {
            span_lint(
                cx,
                ERASING_OP,
                span,
                "this operation will always return zero. This is likely not the intended outcome",
            );
        }
    }
}
