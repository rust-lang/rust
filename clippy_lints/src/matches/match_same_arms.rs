use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::{path_to_local, search_same, SpanlessEq, SpanlessHash};
use rustc_hir::{Arm, Expr, ExprKind, HirId, HirIdMap, HirIdSet, MatchSource, Pat, PatKind};
use rustc_lint::LateContext;
use std::collections::hash_map::Entry;

use super::MATCH_SAME_ARMS;

pub(crate) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>) {
    if let ExprKind::Match(_, arms, MatchSource::Normal) = expr.kind {
        let hash = |&(_, arm): &(usize, &Arm<'_>)| -> u64 {
            let mut h = SpanlessHash::new(cx);
            h.hash_expr(arm.body);
            h.finish()
        };

        let eq = |&(lindex, lhs): &(usize, &Arm<'_>), &(rindex, rhs): &(usize, &Arm<'_>)| -> bool {
            let min_index = usize::min(lindex, rindex);
            let max_index = usize::max(lindex, rindex);

            let mut local_map: HirIdMap<HirId> = HirIdMap::default();
            let eq_fallback = |a: &Expr<'_>, b: &Expr<'_>| {
                if_chain! {
                    if let Some(a_id) = path_to_local(a);
                    if let Some(b_id) = path_to_local(b);
                    let entry = match local_map.entry(a_id) {
                        Entry::Vacant(entry) => entry,
                        // check if using the same bindings as before
                        Entry::Occupied(entry) => return *entry.get() == b_id,
                    };
                    // the names technically don't have to match; this makes the lint more conservative
                    if cx.tcx.hir().name(a_id) == cx.tcx.hir().name(b_id);
                    if cx.typeck_results().expr_ty(a) == cx.typeck_results().expr_ty(b);
                    if pat_contains_local(lhs.pat, a_id);
                    if pat_contains_local(rhs.pat, b_id);
                    then {
                        entry.insert(b_id);
                        true
                    } else {
                        false
                    }
                }
            };
            // Arms with a guard are ignored, those can’t always be merged together
            // This is also the case for arms in-between each there is an arm with a guard
            (min_index..=max_index).all(|index| arms[index].guard.is_none())
                && SpanlessEq::new(cx)
                    .expr_fallback(eq_fallback)
                    .eq_expr(lhs.body, rhs.body)
                // these checks could be removed to allow unused bindings
                && bindings_eq(lhs.pat, local_map.keys().copied().collect())
                && bindings_eq(rhs.pat, local_map.values().copied().collect())
        };

        let indexed_arms: Vec<(usize, &Arm<'_>)> = arms.iter().enumerate().collect();
        for (&(_, i), &(_, j)) in search_same(&indexed_arms, hash, eq) {
            span_lint_and_then(
                cx,
                MATCH_SAME_ARMS,
                j.body.span,
                "this `match` has identical arm bodies",
                |diag| {
                    diag.span_note(i.body.span, "same as this");

                    // Note: this does not use `span_suggestion` on purpose:
                    // there is no clean way
                    // to remove the other arm. Building a span and suggest to replace it to ""
                    // makes an even more confusing error message. Also in order not to make up a
                    // span for the whole pattern, the suggestion is only shown when there is only
                    // one pattern. The user should know about `|` if they are already using it…

                    let lhs = snippet(cx, i.pat.span, "<pat1>");
                    let rhs = snippet(cx, j.pat.span, "<pat2>");

                    if let PatKind::Wild = j.pat.kind {
                        // if the last arm is _, then i could be integrated into _
                        // note that i.pat cannot be _, because that would mean that we're
                        // hiding all the subsequent arms, and rust won't compile
                        diag.span_note(
                            i.body.span,
                            &format!(
                                "`{}` has the same arm body as the `_` wildcard, consider removing it",
                                lhs
                            ),
                        );
                    } else {
                        diag.span_help(i.pat.span, &format!("consider refactoring into `{} | {}`", lhs, rhs,))
                            .help("...or consider changing the match arm bodies");
                    }
                },
            );
        }
    }
}

fn pat_contains_local(pat: &Pat<'_>, id: HirId) -> bool {
    let mut result = false;
    pat.walk_short(|p| {
        result |= matches!(p.kind, PatKind::Binding(_, binding_id, ..) if binding_id == id);
        !result
    });
    result
}

/// Returns true if all the bindings in the `Pat` are in `ids` and vice versa
fn bindings_eq(pat: &Pat<'_>, mut ids: HirIdSet) -> bool {
    let mut result = true;
    pat.each_binding_or_first(&mut |_, id, _, _| result &= ids.remove(&id));
    result && ids.is_empty()
}
