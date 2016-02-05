use regex_syntax;
use std::error::Error;
use syntax::ast::Lit_::LitStr;
use syntax::codemap::{Span, BytePos};
use syntax::parse::token::InternedString;
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
            match_path(path, &REGEX_NEW_PATH) && args.len() == 1
        ], {
            if let ExprLit(ref lit) = args[0].node {
                if let LitStr(ref r, _) = lit.node {
                    if let Err(e) = regex_syntax::Expr::parse(r) {
                        span_lint(cx,
                                  INVALID_REGEX,
                                  str_span(args[0].span, &r, e.position()),
                                  &format!("Regex syntax error: {}",
                                           e.description()));
                    }
                }
            } else {
                if_let_chain!{[
                    let Some(r) = const_str(cx, &*args[0]),
                    let Err(e) = regex_syntax::Expr::parse(&r)
                ], {
                    span_lint(cx,
                              INVALID_REGEX,
                              args[0].span,
                              &format!("Regex syntax error on position {}: {}",
                                       e.position(),
                                       e.description()));
                }}
            }
        }}
    }
}

#[allow(cast_possible_truncation)]
fn str_span(base: Span, s: &str, c: usize) -> Span {
    let lo = match s.char_indices().nth(c) {
        Some((b, _)) => base.lo + BytePos(b as u32),
        _ => base.hi
    };
    Span{ lo: lo, hi: lo, ..base }
}

fn const_str(cx: &LateContext, e: &Expr) -> Option<InternedString> {
    match eval_const_expr_partial(cx.tcx, e, ExprTypeChecked, None) {
        Ok(ConstVal::Str(r)) => Some(r),
        _ => None
    }
}
