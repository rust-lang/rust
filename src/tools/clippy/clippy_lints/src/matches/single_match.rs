use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{expr_block, get_source_text, snippet};
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item, peel_mid_ty_refs};
use clippy_utils::{is_lint_allowed, is_unit_expr, is_wild, peel_blocks, peel_hir_pat_refs, peel_n_hir_expr_refs};
use core::cmp::max;
use rustc_errors::Applicability;
use rustc_hir::{Arm, BindingAnnotation, Block, Expr, ExprKind, Pat, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::{sym, Span};

use super::{MATCH_BOOL, SINGLE_MATCH, SINGLE_MATCH_ELSE};

/// Checks if there are comments contained within a span.
/// This is a very "naive" check, as it just looks for the literal characters // and /* in the
/// source text. This won't be accurate if there are potentially expressions contained within the
/// span, e.g. a string literal `"//"`, but we know that this isn't the case for empty
/// match arms.
fn empty_arm_has_comment(cx: &LateContext<'_>, span: Span) -> bool {
    if let Some(ff) = get_source_text(cx, span)
        && let Some(text) = ff.as_str()
    {
        text.as_bytes().windows(2)
            .any(|w| w == b"//" || w == b"/*")
    } else {
        false
    }
}

#[rustfmt::skip]
pub(crate) fn check(cx: &LateContext<'_>, ex: &Expr<'_>, arms: &[Arm<'_>], expr: &Expr<'_>) {
    if arms.len() == 2 && arms[0].guard.is_none() && arms[1].guard.is_none() {
        if expr.span.from_expansion() {
            // Don't lint match expressions present in
            // macro_rules! block
            return;
        }
        if let PatKind::Or(..) = arms[0].pat.kind {
            // don't lint for or patterns for now, this makes
            // the lint noisy in unnecessary situations
            return;
        }
        let els = arms[1].body;
        let els = if is_unit_expr(peel_blocks(els)) && !empty_arm_has_comment(cx, els.span) {
            None
        } else if let ExprKind::Block(Block { stmts, expr: block_expr, .. }, _) = els.kind {
            if stmts.len() == 1 && block_expr.is_none() || stmts.is_empty() && block_expr.is_some() {
                // single statement/expr "else" block, don't lint
                return;
            }
            // block with 2+ statements or 1 expr and 1+ statement
            Some(els)
        } else {
            // not a block or an emtpy block w/ comments, don't lint
            return;
        };

        let ty = cx.typeck_results().expr_ty(ex);
        if *ty.kind() != ty::Bool || is_lint_allowed(cx, MATCH_BOOL, ex.hir_id) {
            check_single_pattern(cx, ex, arms, expr, els);
            check_opt_like(cx, ex, arms, expr, ty, els);
        }
    }
}

fn check_single_pattern(
    cx: &LateContext<'_>,
    ex: &Expr<'_>,
    arms: &[Arm<'_>],
    expr: &Expr<'_>,
    els: Option<&Expr<'_>>,
) {
    if is_wild(arms[1].pat) {
        report_single_pattern(cx, ex, arms, expr, els);
    }
}

fn report_single_pattern(
    cx: &LateContext<'_>,
    ex: &Expr<'_>,
    arms: &[Arm<'_>],
    expr: &Expr<'_>,
    els: Option<&Expr<'_>>,
) {
    let lint = if els.is_some() { SINGLE_MATCH_ELSE } else { SINGLE_MATCH };
    let ctxt = expr.span.ctxt();
    let mut app = Applicability::HasPlaceholders;
    let els_str = els.map_or(String::new(), |els| {
        format!(" else {}", expr_block(cx, els, ctxt, "..", Some(expr.span), &mut app))
    });

    let (pat, pat_ref_count) = peel_hir_pat_refs(arms[0].pat);
    let (msg, sugg) = if_chain! {
        if let PatKind::Path(_) | PatKind::Lit(_) = pat.kind;
        let (ty, ty_ref_count) = peel_mid_ty_refs(cx.typeck_results().expr_ty(ex));
        if let Some(spe_trait_id) = cx.tcx.lang_items().structural_peq_trait();
        if let Some(pe_trait_id) = cx.tcx.lang_items().eq_trait();
        if ty.is_integral() || ty.is_char() || ty.is_str()
            || (implements_trait(cx, ty, spe_trait_id, &[])
                && implements_trait(cx, ty, pe_trait_id, &[ty.into()]));
        then {
            // scrutinee derives PartialEq and the pattern is a constant.
            let pat_ref_count = match pat.kind {
                // string literals are already a reference.
                PatKind::Lit(Expr { kind: ExprKind::Lit(lit), .. }) if lit.node.is_str() => pat_ref_count + 1,
                _ => pat_ref_count,
            };
            // References are only implicitly added to the pattern, so no overflow here.
            // e.g. will work: match &Some(_) { Some(_) => () }
            // will not: match Some(_) { &Some(_) => () }
            let ref_count_diff = ty_ref_count - pat_ref_count;

            // Try to remove address of expressions first.
            let (ex, removed) = peel_n_hir_expr_refs(ex, ref_count_diff);
            let ref_count_diff = ref_count_diff - removed;

            let msg = "you seem to be trying to use `match` for an equality check. Consider using `if`";
            let sugg = format!(
                "if {} == {}{} {}{els_str}",
                snippet(cx, ex.span, ".."),
                // PartialEq for different reference counts may not exist.
                "&".repeat(ref_count_diff),
                snippet(cx, arms[0].pat.span, ".."),
                expr_block(cx, arms[0].body, ctxt, "..", Some(expr.span), &mut app),
            );
            (msg, sugg)
        } else {
            let msg = "you seem to be trying to use `match` for destructuring a single pattern. Consider using `if let`";
            let sugg = format!(
                "if let {} = {} {}{els_str}",
                snippet(cx, arms[0].pat.span, ".."),
                snippet(cx, ex.span, ".."),
                expr_block(cx, arms[0].body, ctxt, "..", Some(expr.span), &mut app),
            );
            (msg, sugg)
        }
    };

    span_lint_and_sugg(cx, lint, expr.span, msg, "try", sugg, app);
}

fn check_opt_like<'a>(
    cx: &LateContext<'a>,
    ex: &Expr<'_>,
    arms: &[Arm<'_>],
    expr: &Expr<'_>,
    ty: Ty<'a>,
    els: Option<&Expr<'_>>,
) {
    // We don't want to lint if the second arm contains an enum which could
    // have more variants in the future.
    if form_exhaustive_matches(cx, ty, arms[0].pat, arms[1].pat) {
        report_single_pattern(cx, ex, arms, expr, els);
    }
}

/// Returns `true` if all of the types in the pattern are enums which we know
/// won't be expanded in the future
fn pat_in_candidate_enum<'a>(cx: &LateContext<'a>, ty: Ty<'a>, pat: &Pat<'_>) -> bool {
    let mut paths_and_types = Vec::new();
    collect_pat_paths(&mut paths_and_types, cx, pat, ty);
    paths_and_types.iter().all(|ty| in_candidate_enum(cx, *ty))
}

/// Returns `true` if the given type is an enum we know won't be expanded in the future
fn in_candidate_enum(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    // list of candidate `Enum`s we know will never get any more members
    let candidates = [sym::Cow, sym::Option, sym::Result];

    for candidate_ty in candidates {
        if is_type_diagnostic_item(cx, ty, candidate_ty) {
            return true;
        }
    }
    false
}

/// Collects types from the given pattern
fn collect_pat_paths<'a>(acc: &mut Vec<Ty<'a>>, cx: &LateContext<'a>, pat: &Pat<'_>, ty: Ty<'a>) {
    match pat.kind {
        PatKind::Tuple(inner, _) => inner.iter().for_each(|p| {
            let p_ty = cx.typeck_results().pat_ty(p);
            collect_pat_paths(acc, cx, p, p_ty);
        }),
        PatKind::TupleStruct(..) | PatKind::Binding(BindingAnnotation::NONE, .., None) | PatKind::Path(_) => {
            acc.push(ty);
        },
        _ => {},
    }
}

/// Returns true if the given arm of pattern matching contains wildcard patterns.
fn contains_only_wilds(pat: &Pat<'_>) -> bool {
    match pat.kind {
        PatKind::Wild => true,
        PatKind::Tuple(inner, _) | PatKind::TupleStruct(_, inner, ..) => inner.iter().all(contains_only_wilds),
        _ => false,
    }
}

/// Returns true if the given patterns forms only exhaustive matches that don't contain enum
/// patterns without a wildcard.
fn form_exhaustive_matches<'a>(cx: &LateContext<'a>, ty: Ty<'a>, left: &Pat<'_>, right: &Pat<'_>) -> bool {
    match (&left.kind, &right.kind) {
        (PatKind::Wild, _) | (_, PatKind::Wild) => true,
        (PatKind::Tuple(left_in, left_pos), PatKind::Tuple(right_in, right_pos)) => {
            // We don't actually know the position and the presence of the `..` (dotdot) operator
            // in the arms, so we need to evaluate the correct offsets here in order to iterate in
            // both arms at the same time.
            let left_pos = left_pos.as_opt_usize();
            let right_pos = right_pos.as_opt_usize();
            let len = max(
                left_in.len() + usize::from(left_pos.is_some()),
                right_in.len() + usize::from(right_pos.is_some()),
            );
            let mut left_pos = left_pos.unwrap_or(usize::MAX);
            let mut right_pos = right_pos.unwrap_or(usize::MAX);
            let mut left_dot_space = 0;
            let mut right_dot_space = 0;
            for i in 0..len {
                let mut found_dotdot = false;
                if i == left_pos {
                    left_dot_space += 1;
                    if left_dot_space < len - left_in.len() {
                        left_pos += 1;
                    }
                    found_dotdot = true;
                }
                if i == right_pos {
                    right_dot_space += 1;
                    if right_dot_space < len - right_in.len() {
                        right_pos += 1;
                    }
                    found_dotdot = true;
                }
                if found_dotdot {
                    continue;
                }
                if !contains_only_wilds(&left_in[i - left_dot_space])
                    && !contains_only_wilds(&right_in[i - right_dot_space])
                {
                    return false;
                }
            }
            true
        },
        (PatKind::TupleStruct(..), PatKind::Path(_)) => pat_in_candidate_enum(cx, ty, right),
        (PatKind::TupleStruct(..), PatKind::TupleStruct(_, inner, _)) => {
            pat_in_candidate_enum(cx, ty, right) && inner.iter().all(contains_only_wilds)
        },
        _ => false,
    }
}
