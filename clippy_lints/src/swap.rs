use rustc::lint::*;
use rustc::hir::*;
use syntax::codemap::mk_sp;
use utils::{differing_macro_contexts, snippet_opt, span_lint_and_then, SpanlessEq};

/// **What it does:** This lints manual swapping.
///
/// **Why is this bad?** The `std::mem::swap` function exposes the intent better without
/// deinitializing or copying either variable.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust,ignore
/// let t = b;
/// b = a;
/// a = t;
/// ```
declare_lint! {
    pub MANUAL_SWAP,
    Warn,
    "manual swap"
}

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
    pub ALMOST_SWAPPED,
    Warn,
    "`foo = bar; bar = foo` sequence"
}

#[derive(Copy,Clone)]
pub struct Swap;

impl LintPass for Swap {
    fn get_lints(&self) -> LintArray {
        lint_array![MANUAL_SWAP, ALMOST_SWAPPED]
    }
}

impl LateLintPass for Swap {
    fn check_block(&mut self, cx: &LateContext, block: &Block) {
        check_manual_swap(cx, block);
        check_suspicious_swap(cx, block);
    }
}

/// Implementation of the `MANUAL_SWAP` lint.
fn check_manual_swap(cx: &LateContext, block: &Block) {
    for w in block.stmts.windows(3) {
        if_let_chain!{[
            // let t = foo();
            let StmtDecl(ref tmp, _) = w[0].node,
            let DeclLocal(ref tmp) = tmp.node,
            let Some(ref tmp_init) = tmp.init,
            let PatKind::Binding(_, ref tmp_name, None) = tmp.pat.node,

            // foo() = bar();
            let StmtSemi(ref first, _) = w[1].node,
            let ExprAssign(ref lhs1, ref rhs1) = first.node,

            // bar() = t;
            let StmtSemi(ref second, _) = w[2].node,
            let ExprAssign(ref lhs2, ref rhs2) = second.node,
            let ExprPath(None, ref rhs2) = rhs2.node,
            rhs2.segments.len() == 1,

            tmp_name.node.as_str() == rhs2.segments[0].name.as_str(),
            SpanlessEq::new(cx).ignore_fn().eq_expr(tmp_init, lhs1),
            SpanlessEq::new(cx).ignore_fn().eq_expr(rhs1, lhs2)
        ], {
            let (what, lhs, rhs) = if let (Some(first), Some(second)) = (snippet_opt(cx, lhs1.span), snippet_opt(cx, rhs1.span)) {
                (format!(" `{}` and `{}`", first, second), first, second)
            } else {
                ("".to_owned(), "".to_owned(), "".to_owned())
            };

            let span = mk_sp(tmp.span.lo, second.span.hi);

            span_lint_and_then(cx,
                               MANUAL_SWAP,
                               span,
                               &format!("this looks like you are swapping{} manually", what),
                               |db| {
                                   if !what.is_empty() {
                                       db.span_suggestion(span, "try",
                                                          format!("std::mem::swap(&mut {}, &mut {})", lhs, rhs));
                                       db.note("or maybe you should use `std::mem::replace`?");
                                   }
                               });
        }}
    }
}

/// Implementation of the `ALMOST_SWAPPED` lint.
fn check_suspicious_swap(cx: &LateContext, block: &Block) {
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
                               ALMOST_SWAPPED,
                               span,
                               &format!("this looks like you are trying to swap{}", what),
                               |db| {
                                   if !what.is_empty() {
                                       db.span_suggestion(span, "try",
                                                          format!("std::mem::swap(&mut {}, &mut {})", lhs, rhs));
                                       db.note("or maybe you should use `std::mem::replace`?");
                                   }
                               });
        }}
    }
}
