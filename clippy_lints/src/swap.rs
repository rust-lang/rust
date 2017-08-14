use rustc::hir::*;
use rustc::lint::*;
use rustc::ty;
use utils::{differing_macro_contexts, match_type, paths, snippet, span_lint_and_then, walk_ptrs_ty, SpanlessEq};
use utils::sugg::Sugg;
use syntax_pos::{Span, NO_EXPANSION};

/// **What it does:** Checks for manual swapping.
///
/// **Why is this bad?** The `std::mem::swap` function exposes the intent better
/// without deinitializing or copying either variable.
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
    "manual swap of two variables"
}

/// **What it does:** Checks for `foo = bar; bar = foo` sequences.
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

#[derive(Copy, Clone)]
pub struct Swap;

impl LintPass for Swap {
    fn get_lints(&self) -> LintArray {
        lint_array![MANUAL_SWAP, ALMOST_SWAPPED]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Swap {
    fn check_block(&mut self, cx: &LateContext<'a, 'tcx>, block: &'tcx Block) {
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
            let PatKind::Binding(_, _, ref tmp_name, None) = tmp.pat.node,

            // foo() = bar();
            let StmtSemi(ref first, _) = w[1].node,
            let ExprAssign(ref lhs1, ref rhs1) = first.node,

            // bar() = t;
            let StmtSemi(ref second, _) = w[2].node,
            let ExprAssign(ref lhs2, ref rhs2) = second.node,
            let ExprPath(QPath::Resolved(None, ref rhs2)) = rhs2.node,
            rhs2.segments.len() == 1,

            tmp_name.node.as_str() == rhs2.segments[0].name.as_str(),
            SpanlessEq::new(cx).ignore_fn().eq_expr(tmp_init, lhs1),
            SpanlessEq::new(cx).ignore_fn().eq_expr(rhs1, lhs2)
        ], {
            fn check_for_slice<'a>(
                cx: &LateContext,
                lhs1: &'a Expr,
                lhs2: &'a Expr,
            ) -> Option<(&'a Expr, &'a Expr, &'a Expr)> {
                if let ExprIndex(ref lhs1, ref idx1) = lhs1.node {
                    if let ExprIndex(ref lhs2, ref idx2) = lhs2.node {
                        if SpanlessEq::new(cx).ignore_fn().eq_expr(lhs1, lhs2) {
                            let ty = walk_ptrs_ty(cx.tables.expr_ty(lhs1));

                            if matches!(ty.sty, ty::TySlice(_)) ||
                                matches!(ty.sty, ty::TyArray(_, _)) ||
                                match_type(cx, ty, &paths::VEC) ||
                                match_type(cx, ty, &paths::VEC_DEQUE) {
                                    return Some((lhs1, idx1, idx2));
                            }
                        }
                    }
                }

                None
            }

            let (replace, what, sugg) = if let Some((slice, idx1, idx2)) = check_for_slice(cx, lhs1, lhs2) {
                if let Some(slice) = Sugg::hir_opt(cx, slice) {
                    (false,
                     format!(" elements of `{}`", slice),
                     format!("{}.swap({}, {})",
                             slice.maybe_par(),
                             snippet(cx, idx1.span, ".."),
                             snippet(cx, idx2.span, "..")))
                } else {
                    (false, "".to_owned(), "".to_owned())
                }
            } else if let (Some(first), Some(second)) = (Sugg::hir_opt(cx, lhs1), Sugg::hir_opt(cx, rhs1)) {
                (true, format!(" `{}` and `{}`", first, second),
                    format!("std::mem::swap({}, {})", first.mut_addr(), second.mut_addr()))
            } else {
                (true, "".to_owned(), "".to_owned())
            };

            let span = Span { lo: w[0].span.lo, hi: second.span.hi, ctxt: NO_EXPANSION};

            span_lint_and_then(cx,
                               MANUAL_SWAP,
                               span,
                               &format!("this looks like you are swapping{} manually", what),
                               |db| {
                                   if !sugg.is_empty() {
                                       db.span_suggestion(span, "try", sugg);

                                       if replace {
                                           db.note("or maybe you should use `std::mem::replace`?");
                                       }
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
            let lhs0 = Sugg::hir_opt(cx, lhs0);
            let rhs0 = Sugg::hir_opt(cx, rhs0);
            let (what, lhs, rhs) = if let (Some(first), Some(second)) = (lhs0, rhs0) {
                (format!(" `{}` and `{}`", first, second), first.mut_addr().to_string(), second.mut_addr().to_string())
            } else {
                ("".to_owned(), "".to_owned(), "".to_owned())
            };

            let span = Span{ lo: first.span.lo, hi: second.span.hi, ctxt: NO_EXPANSION};

            span_lint_and_then(cx,
                               ALMOST_SWAPPED,
                               span,
                               &format!("this looks like you are trying to swap{}", what),
                               |db| {
                                   if !what.is_empty() {
                                       db.span_suggestion(span, "try",
                                                          format!("std::mem::swap({}, {})", lhs, rhs));
                                       db.note("or maybe you should use `std::mem::replace`?");
                                   }
                               });
        }}
    }
}
