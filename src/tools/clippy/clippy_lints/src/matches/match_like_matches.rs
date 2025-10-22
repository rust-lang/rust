//! Lint a `match` or `if let .. { .. } else { .. }` expr that could be replaced by `matches!`

use super::REDUNDANT_PATTERN_MATCHING;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{is_lint_allowed, is_wild, span_contains_comment};
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Arm, BorrowKind, Expr, ExprKind, Pat, PatKind, QPath};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::ty;
use rustc_span::source_map::Spanned;

use super::MATCH_LIKE_MATCHES_MACRO;

pub(crate) fn check_if_let<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    let_pat: &'tcx Pat<'_>,
    let_expr: &'tcx Expr<'_>,
    then_expr: &'tcx Expr<'_>,
    else_expr: &'tcx Expr<'_>,
) {
    if !span_contains_comment(cx.sess().source_map(), expr.span)
        && cx.typeck_results().expr_ty(expr).is_bool()
        && let Some(b0) = find_bool_lit(then_expr)
        && let Some(b1) = find_bool_lit(else_expr)
        && b0 != b1
    {
        if !is_lint_allowed(cx, REDUNDANT_PATTERN_MATCHING, let_pat.hir_id) && is_some_wild(let_pat.kind) {
            return;
        }

        // The suggestion may be incorrect, because some arms can have `cfg` attributes
        // evaluated into `false` and so such arms will be stripped before.
        let mut applicability = Applicability::MaybeIncorrect;
        let pat = snippet_with_applicability(cx, let_pat.span, "..", &mut applicability);

        // strip potential borrows (#6503), but only if the type is a reference
        let mut ex_new = let_expr;
        if let ExprKind::AddrOf(BorrowKind::Ref, .., ex_inner) = let_expr.kind
            && let ty::Ref(..) = cx.typeck_results().expr_ty(ex_inner).kind()
        {
            ex_new = ex_inner;
        }
        span_lint_and_sugg(
            cx,
            MATCH_LIKE_MATCHES_MACRO,
            expr.span,
            "if let .. else expression looks like `matches!` macro",
            "try",
            format!(
                "{}matches!({}, {pat})",
                if b0 { "" } else { "!" },
                snippet_with_applicability(cx, ex_new.span, "..", &mut applicability),
            ),
            applicability,
        );
    }
}

pub(super) fn check_match<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    scrutinee: &'tcx Expr<'_>,
    arms: &'tcx [Arm<'tcx>],
) -> bool {
    if let Some((last_arm, arms_without_last)) = arms.split_last()
        && let Some((first_arm, middle_arms)) = arms_without_last.split_first()
        && !span_contains_comment(cx.sess().source_map(), e.span)
        && cx.typeck_results().expr_ty(e).is_bool()
        && let Some(b0) = find_bool_lit(first_arm.body)
        && let Some(b1) = find_bool_lit(last_arm.body)
        && b0 != b1
        // We handle two cases:
        && (
            // - There are no middle arms, i.e., 2 arms in total
            //
            // In that case, the first arm may or may not have a guard, because this:
            // ```rs
            // match e {
            //     Either::Left $(if $guard)|+ => true, // or `false`, but then we'll need `!matches!(..)`
            //     _ => false,
            // }
            // ```
            // can always become this:
            // ```rs
            // matches!(e, Either::Left $(if $guard)|+)
            // ```
            middle_arms.is_empty()

            // - (added in #6216) There are middle arms
            //
            // In that case, neither they nor the first arm may have guards
            // -- otherwise, they couldn't be combined into an or-pattern in `matches!`
            //
            // This:
            // ```rs
            // match e {
            //     Either3::First => true,
            //     Either3::Second => true,
            //     _ /* matches `Either3::Third` */ => false,
            // }
            // ```
            // can become this:
            // ```rs
            // matches!(e, Either3::First | Either3::Second)
            // ```
            //
            // But this:
            // ```rs
            // match e {
            //     Either3::First if X => true,
            //     Either3::Second => true,
            //     _ => false,
            // }
            // ```
            // cannot be transformed.
            //
            // We set an additional constraint of all of them needing to return the same bool,
            // so we don't lint things like:
            // ```rs
            // match e {
            //     Either3::First => true,
            //     Either3::Second => false,
            //     _ => false,
            // }
            // ```
            // This is not *strictly* necessary, but it simplifies the logic a bit
            || arms_without_last.iter().all(|arm| {
                cx.tcx.hir_attrs(arm.hir_id).is_empty() && arm.guard.is_none() && find_bool_lit(arm.body) == Some(b0)
            })
        )
    {
        if !is_wild(last_arm.pat) {
            return false;
        }

        for arm in arms_without_last {
            let pat = arm.pat;
            if !is_lint_allowed(cx, REDUNDANT_PATTERN_MATCHING, pat.hir_id) && is_some_wild(pat.kind) {
                return false;
            }
        }

        // The suggestion may be incorrect, because some arms can have `cfg` attributes
        // evaluated into `false` and so such arms will be stripped before.
        let mut applicability = Applicability::MaybeIncorrect;
        let pat = {
            use itertools::Itertools as _;
            arms_without_last
                .iter()
                .map(|arm| snippet_with_applicability(cx, arm.pat.span, "..", &mut applicability))
                .join(" | ")
        };
        let pat_and_guard = if let Some(g) = first_arm.guard {
            format!(
                "{pat} if {}",
                snippet_with_applicability(cx, g.span, "..", &mut applicability)
            )
        } else {
            pat
        };

        // strip potential borrows (#6503), but only if the type is a reference
        let mut ex_new = scrutinee;
        if let ExprKind::AddrOf(BorrowKind::Ref, .., ex_inner) = scrutinee.kind
            && let ty::Ref(..) = cx.typeck_results().expr_ty(ex_inner).kind()
        {
            ex_new = ex_inner;
        }
        span_lint_and_sugg(
            cx,
            MATCH_LIKE_MATCHES_MACRO,
            e.span,
            "match expression looks like `matches!` macro",
            "try",
            format!(
                "{}matches!({}, {pat_and_guard})",
                if b0 { "" } else { "!" },
                snippet_with_applicability(cx, ex_new.span, "..", &mut applicability),
            ),
            applicability,
        );
        true
    } else {
        false
    }
}

/// Extract a `bool` or `{ bool }`
fn find_bool_lit(ex: &Expr<'_>) -> Option<bool> {
    match ex.kind {
        ExprKind::Lit(Spanned {
            node: LitKind::Bool(b), ..
        }) => Some(b),
        ExprKind::Block(
            rustc_hir::Block {
                stmts: [],
                expr: Some(exp),
                ..
            },
            _,
        ) => {
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

/// Checks whether a pattern is `Some(_)`
fn is_some_wild(pat_kind: PatKind<'_>) -> bool {
    match pat_kind {
        PatKind::TupleStruct(QPath::Resolved(_, path), [first, ..], _) if is_wild(first) => {
            let name = path.segments[0].ident;
            name.name == rustc_span::sym::Some
        },
        _ => false,
    }
}
