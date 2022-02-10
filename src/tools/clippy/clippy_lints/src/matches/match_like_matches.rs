use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{higher, is_wild};
use rustc_ast::{Attribute, LitKind};
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, Guard, MatchSource, Pat};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::source_map::Spanned;

use super::MATCH_LIKE_MATCHES_MACRO;

/// Lint a `match` or `if let .. { .. } else { .. }` expr that could be replaced by `matches!`
pub(crate) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> bool {
    if let Some(higher::IfLet {
        let_pat,
        let_expr,
        if_then,
        if_else: Some(if_else),
    }) = higher::IfLet::hir(cx, expr)
    {
        return find_matches_sugg(
            cx,
            let_expr,
            IntoIterator::into_iter([(&[][..], Some(let_pat), if_then, None), (&[][..], None, if_else, None)]),
            expr,
            true,
        );
    }

    if let ExprKind::Match(scrut, arms, MatchSource::Normal) = expr.kind {
        return find_matches_sugg(
            cx,
            scrut,
            arms.iter().map(|arm| {
                (
                    cx.tcx.hir().attrs(arm.hir_id),
                    Some(arm.pat),
                    arm.body,
                    arm.guard.as_ref(),
                )
            }),
            expr,
            false,
        );
    }

    false
}

/// Lint a `match` or `if let` for replacement by `matches!`
fn find_matches_sugg<'a, 'b, I>(
    cx: &LateContext<'_>,
    ex: &Expr<'_>,
    mut iter: I,
    expr: &Expr<'_>,
    is_if_let: bool,
) -> bool
where
    'b: 'a,
    I: Clone
        + DoubleEndedIterator
        + ExactSizeIterator
        + Iterator<
            Item = (
                &'a [Attribute],
                Option<&'a Pat<'b>>,
                &'a Expr<'b>,
                Option<&'a Guard<'b>>,
            ),
        >,
{
    if_chain! {
        if iter.len() >= 2;
        if cx.typeck_results().expr_ty(expr).is_bool();
        if let Some((_, last_pat_opt, last_expr, _)) = iter.next_back();
        let iter_without_last = iter.clone();
        if let Some((first_attrs, _, first_expr, first_guard)) = iter.next();
        if let Some(b0) = find_bool_lit(&first_expr.kind, is_if_let);
        if let Some(b1) = find_bool_lit(&last_expr.kind, is_if_let);
        if b0 != b1;
        if first_guard.is_none() || iter.len() == 0;
        if first_attrs.is_empty();
        if iter
            .all(|arm| {
                find_bool_lit(&arm.2.kind, is_if_let).map_or(false, |b| b == b0) && arm.3.is_none() && arm.0.is_empty()
            });
        then {
            if let Some(last_pat) = last_pat_opt {
                if !is_wild(last_pat) {
                    return false;
                }
            }

            // The suggestion may be incorrect, because some arms can have `cfg` attributes
            // evaluated into `false` and so such arms will be stripped before.
            let mut applicability = Applicability::MaybeIncorrect;
            let pat = {
                use itertools::Itertools as _;
                iter_without_last
                    .filter_map(|arm| {
                        let pat_span = arm.1?.span;
                        Some(snippet_with_applicability(cx, pat_span, "..", &mut applicability))
                    })
                    .join(" | ")
            };
            let pat_and_guard = if let Some(Guard::If(g)) = first_guard {
                format!("{} if {}", pat, snippet_with_applicability(cx, g.span, "..", &mut applicability))
            } else {
                pat
            };

            // strip potential borrows (#6503), but only if the type is a reference
            let mut ex_new = ex;
            if let ExprKind::AddrOf(BorrowKind::Ref, .., ex_inner) = ex.kind {
                if let ty::Ref(..) = cx.typeck_results().expr_ty(ex_inner).kind() {
                    ex_new = ex_inner;
                }
            };
            span_lint_and_sugg(
                cx,
                MATCH_LIKE_MATCHES_MACRO,
                expr.span,
                &format!("{} expression looks like `matches!` macro", if is_if_let { "if let .. else" } else { "match" }),
                "try this",
                format!(
                    "{}matches!({}, {})",
                    if b0 { "" } else { "!" },
                    snippet_with_applicability(cx, ex_new.span, "..", &mut applicability),
                    pat_and_guard,
                ),
                applicability,
            );
            true
        } else {
            false
        }
    }
}

/// Extract a `bool` or `{ bool }`
fn find_bool_lit(ex: &ExprKind<'_>, is_if_let: bool) -> Option<bool> {
    match ex {
        ExprKind::Lit(Spanned {
            node: LitKind::Bool(b), ..
        }) => Some(*b),
        ExprKind::Block(
            rustc_hir::Block {
                stmts: &[],
                expr: Some(exp),
                ..
            },
            _,
        ) if is_if_let => {
            if let ExprKind::Lit(Spanned {
                node: LitKind::Bool(b), ..
            }) = exp.kind
            {
                Some(b)
            } else {
                None
            }
        },
        _ => None,
    }
}
