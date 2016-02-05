use rustc::lint::*;
use rustc_front::hir::*;
use utils::{IO_PRINT_PATH, is_expn_of, match_path, span_lint};

/// **What it does:** This lint warns whenever you print on *stdout*. The purpose of this lint is to catch debugging remnants.
///
/// **Why is this bad?** People often print on *stdout* while debugging an application and might
/// forget to remove those prints afterward.
///
/// **Known problems:** Only catches `print!` and `println!` calls.
///
/// **Example:** `println!("Hello world!");`
declare_lint! {
    pub PRINT_STDOUT,
    Allow,
    "printing on stdout"
}

#[derive(Copy, Clone, Debug)]
pub struct PrintLint;

impl LintPass for PrintLint {
    fn get_lints(&self) -> LintArray {
        lint_array!(PRINT_STDOUT)
    }
}

impl LateLintPass for PrintLint {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if let ExprCall(ref fun, _) = expr.node {
            if let ExprPath(_, ref path) = fun.node {
                if match_path(path, &IO_PRINT_PATH) {
                    if let Some(span) = is_expn_of(cx, expr.span, "print") {
                        let (span, name) = match is_expn_of(cx, span, "println") {
                            Some(span) => (span, "println"),
                            None => (span, "print"),
                        };

                        span_lint(cx, PRINT_STDOUT, span, &format!("use of `{}!`", name));
                    }
                }
            }
        }
    }
}
