use rustc::lint::*;
use syntax::codemap::mk_sp;
use syntax::ast;
use utils::{differing_macro_contexts, in_macro, snippet_opt, span_note_and_lint};
use syntax::ptr::P;

/// **What it does:** This lint looks for use of the non-existent `=*`, `=!` and `=-` operators.
///
/// **Why is this bad?** This either a typo of `*=`, `!=` or `-=` or confusing.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust,ignore
/// a =- 42; // confusing, should it be `a -= 42` or `a = -42`?
/// ```
declare_lint! {
    pub SUSPICIOUS_ASSIGNMENT_FORMATTING,
    Warn,
    "suspicious formatting of `*=`, `-=` or `!=`"
}

#[derive(Copy,Clone)]
pub struct Formatting;

impl LintPass for Formatting {
    fn get_lints(&self) -> LintArray {
        lint_array![SUSPICIOUS_ASSIGNMENT_FORMATTING]
    }
}

impl EarlyLintPass for Formatting {
    fn check_expr(&mut self, cx: &EarlyContext, expr: &ast::Expr) {
        check_assign(cx, expr);
    }
}

/// Implementation of the SUSPICIOUS_ASSIGNMENT_FORMATTING lint.
fn check_assign(cx: &EarlyContext, expr: &ast::Expr) {
    if let ast::ExprKind::Assign(ref lhs, ref rhs) = expr.node {
        if !differing_macro_contexts(lhs.span, rhs.span) && !in_macro(cx, lhs.span) {
            let eq_span = mk_sp(lhs.span.hi, rhs.span.lo);

            if let Some((sub_rhs, op)) = check_unop(rhs) {
                if let Some(eq_snippet) = snippet_opt(cx, eq_span) {
                    let eqop_span = mk_sp(lhs.span.hi, sub_rhs.span.lo);
                    if eq_snippet.ends_with('=') {
                        span_note_and_lint(cx,
                                           SUSPICIOUS_ASSIGNMENT_FORMATTING,
                                           eqop_span,
                                           &format!("this looks like you are trying to use `.. {op}= ..`, but you really are doing `.. = ({op} ..)`", op=op),
                                           eqop_span,
                                           &format!("to remove this lint, use either `{op}=` or `= {op}`", op=op));
                    }
                }
            }
        }
    }
}

fn check_unop(expr: &ast::Expr) -> Option<(&P<ast::Expr>, &'static str)> {
    match expr.node {
        ast::ExprKind::Unary(op, ref expr) => Some((expr, ast::UnOp::to_string(op))),
        _ => None,
    }
}
