use rustc::lint::*;
use rustc_front::hir::*;
use utils::{differing_macro_contexts, snippet_opt, span_lint_and_then, SpanlessEq};
use syntax::codemap::mk_sp;

/// **What it does:** This lints `foo = bar; bar = foo` sequences.
///
/// **Why is this bad?** This looks like a failed attempt to swap.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust,ignore
/// a = b;
/// b = a;
/// ```
declare_lint! {
    pub SUSPICIOUS_SWAP,
    Warn,
    "`foo = bar; bar = foo` sequence"
}

#[derive(Copy,Clone)]
pub struct Swap;

impl LintPass for Swap {
    fn get_lints(&self) -> LintArray {
        lint_array![SUSPICIOUS_SWAP]
    }
}

impl LateLintPass for Swap {
    fn check_block(&mut self, cx: &LateContext, block: &Block) {
        for w in block.stmts.windows(2) {
            if_let_chain!{[
                let StmtSemi(ref first, _) = w[0].node,
                let StmtSemi(ref second, _) = w[1].node,
                !differing_macro_contexts(first.span, second.span),
                let ExprAssign(ref lhs0, ref rhs0) = first.node,
                let ExprAssign(ref lhs1, ref rhs1) = second.node,
                SpanlessEq::new(cx).ignore_fn().eq_expr(lhs0, rhs1),
                SpanlessEq::new(cx).ignore_fn().eq_expr(lhs1, rhs0)
            ], {
                let (what, lhs, rhs) = if let (Some(first), Some(second)) = (snippet_opt(cx, lhs0.span), snippet_opt(cx, rhs0.span)) {
                    (format!(" `{}` and `{}`", first, second), first, second)
                } else {
                    ("".to_owned(), "".to_owned(), "".to_owned())
                };

                let span = mk_sp(first.span.lo, second.span.hi);

                span_lint_and_then(cx,
                                   SUSPICIOUS_SWAP,
                                   span,
                                   &format!("this looks like you are trying to swap{}", what),
                                   |db| {
                                       if !what.is_empty() {
                                           db.span_suggestion(span, "try",
                                                              format!("std::mem::swap({}, {})", lhs, rhs));
                                           db.fileline_note(span, "or maybe you should use `std::mem::replace`?");
                                       }
                                   });
            }}
        }
    }
}
