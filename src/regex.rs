use regex_syntax;
use std::error::Error;
use syntax::codemap::{Span, BytePos, Pos};
use rustc_front::hir::*;
use rustc::middle::const_eval::{eval_const_expr_partial, ConstVal};
use rustc::middle::const_eval::EvalHint::ExprTypeChecked;
use rustc::lint::*;

use utils::{match_path, REGEX_NEW_PATH, span_lint};

/// **What it does:** This lint checks `Regex::new(_)` invocations for correct regex syntax. It is `deny` by default.
///
/// **Why is this bad?** This will lead to a runtime panic.
///
/// **Known problems:** None.
///
/// **Example:** `Regex::new("|")`
declare_lint! {
    pub INVALID_REGEX,
    Deny,
    "finds invalid regular expressions in `Regex::new(_)` invocations"
}

#[derive(Copy,Clone)]
pub struct RegexPass;

impl LintPass for RegexPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(INVALID_REGEX)
    }
}

impl LateLintPass for RegexPass {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if_let_chain!{[
            let ExprCall(ref fun, ref args) = expr.node,
            let ExprPath(_, ref path) = fun.node,
            match_path(path, &REGEX_NEW_PATH) && args.len() == 1,
            let Ok(ConstVal::Str(r)) = eval_const_expr_partial(cx.tcx, 
                                                               &*args[0],
                                                               ExprTypeChecked,
                                                               None),
            let Err(e) = regex_syntax::Expr::parse(&r)
        ], {
            let lo = args[0].span.lo + BytePos::from_usize(e.position());
            let span = Span{ lo: lo, hi: lo, expn_id: args[0].span.expn_id };
            span_lint(cx,
                      INVALID_REGEX,
                      span,
                      &format!("Regex syntax error: {}", e.description()));
        }}
    }
}
