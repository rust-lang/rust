use rustc_middle::mir;
use rustc_middle::mir::coverage::{Mapping, MappingKind, START_BCB};
use rustc_middle::ty::TyCtxt;
use rustc_span::source_map::SourceMap;
use rustc_span::{BytePos, DesugaringKind, ExpnId, ExpnKind, MacroKind, Span};
use tracing::instrument;

use crate::coverage::expansion::{self, ExpnTree, SpanWithBcb};
use crate::coverage::graph::{BasicCoverageBlock, CoverageGraph};
use crate::coverage::hir_info::ExtractedHirInfo;
use crate::coverage::spans::from_mir::{Hole, RawSpanFromMir};

mod from_mir;

pub(super) fn extract_refined_covspans<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    hir_info: &ExtractedHirInfo,
    graph: &CoverageGraph,
    mappings: &mut Vec<Mapping>,
) {
    if hir_info.is_async_fn {
        // An async function desugars into a function that returns a future,
        // with the user code wrapped in a closure. Any spans in the desugared
        // outer function will be unhelpful, so just keep the signature span
        // and ignore all of the spans in the MIR body.
        if let Some(span) = hir_info.fn_sig_span {
            mappings.push(Mapping { span, kind: MappingKind::Code { bcb: START_BCB } })
        }
        return;
    }

    let &ExtractedHirInfo { body_span, .. } = hir_info;

    let raw_spans = from_mir::extract_raw_spans_from_mir(mir_body, graph);
    // Use the raw spans to build a tree of expansions for this function.
    let expn_tree = expansion::build_expn_tree(
        raw_spans
            .into_iter()
            .map(|RawSpanFromMir { raw_span, bcb }| SpanWithBcb { span: raw_span, bcb }),
    );

    let mut covspans = vec![];
    let mut push_covspan = |covspan: Covspan| {
        let covspan_span = covspan.span;
        // Discard any spans not contained within the function body span.
        // Also discard any spans that fill the entire body, because they tend
        // to represent compiler-inserted code, e.g. implicitly returning `()`.
        if !body_span.contains(covspan_span) || body_span.source_equal(covspan_span) {
            return;
        }

        // Each pushed covspan should have the same context as the body span.
        // If it somehow doesn't, discard the covspan, or panic in debug builds.
        if !body_span.eq_ctxt(covspan_span) {
            debug_assert!(
                false,
                "span context mismatch: body_span={body_span:?}, covspan.span={covspan_span:?}"
            );
            return;
        }

        covspans.push(covspan);
    };

    if let Some(node) = expn_tree.get(body_span.ctxt().outer_expn()) {
        for &SpanWithBcb { span, bcb } in &node.spans {
            push_covspan(Covspan { span, bcb });
        }

        // For each expansion with its call-site in the body span, try to
        // distill a corresponding covspan.
        for &child_expn_id in &node.child_expn_ids {
            if let Some(covspan) =
                single_covspan_for_child_expn(tcx, graph, &expn_tree, child_expn_id)
            {
                push_covspan(covspan);
            }
        }
    }

    // Only proceed if we found at least one usable span.
    if covspans.is_empty() {
        return;
    }

    // Also add the function signature span, if available.
    // Otherwise, add a fake span at the start of the body, to avoid an ugly
    // gap between the start of the body and the first real span.
    // FIXME: Find a more principled way to solve this problem.
    covspans.push(Covspan {
        span: hir_info.fn_sig_span.unwrap_or_else(|| body_span.shrink_to_lo()),
        bcb: START_BCB,
    });

    let compare_covspans = |a: &Covspan, b: &Covspan| {
        compare_spans(a.span, b.span)
            // After deduplication, we want to keep only the most-dominated BCB.
            .then_with(|| graph.cmp_in_dominator_order(a.bcb, b.bcb).reverse())
    };
    covspans.sort_by(compare_covspans);

    // Among covspans with the same span, keep only one,
    // preferring the one with the most-dominated BCB.
    // (Ideally we should try to preserve _all_ non-dominating BCBs, but that
    // requires a lot more complexity in the span refiner, for little benefit.)
    covspans.dedup_by(|b, a| a.span.source_equal(b.span));

    // Sort the holes, and merge overlapping/adjacent holes.
    let mut holes = hir_info
        .hole_spans
        .iter()
        .copied()
        // Discard any holes that aren't directly visible within the body span.
        .filter(|&hole_span| body_span.contains(hole_span) && body_span.eq_ctxt(hole_span))
        .map(|span| Hole { span })
        .collect::<Vec<_>>();
    holes.sort_by(|a, b| compare_spans(a.span, b.span));
    holes.dedup_by(|b, a| a.merge_if_overlapping_or_adjacent(b));

    // Discard any span that overlaps with a hole.
    discard_spans_overlapping_holes(&mut covspans, &holes);

    // Discard spans that overlap in unwanted ways.
    let mut covspans = remove_unwanted_overlapping_spans(covspans);

    // For all empty spans, either enlarge them to be non-empty, or discard them.
    let source_map = tcx.sess.source_map();
    covspans.retain_mut(|covspan| {
        let Some(span) = ensure_non_empty_span(source_map, covspan.span) else { return false };
        covspan.span = span;
        true
    });

    // Merge covspans that can be merged.
    covspans.dedup_by(|b, a| a.merge_if_eligible(b));

    mappings.extend(covspans.into_iter().map(|Covspan { span, bcb }| {
        // Each span produced by the refiner represents an ordinary code region.
        Mapping { span, kind: MappingKind::Code { bcb } }
    }));
}

/// For a single child expansion, try to distill it into a single span+BCB mapping.
fn single_covspan_for_child_expn(
    tcx: TyCtxt<'_>,
    graph: &CoverageGraph,
    expn_tree: &ExpnTree,
    expn_id: ExpnId,
) -> Option<Covspan> {
    let node = expn_tree.get(expn_id)?;

    let bcbs =
        expn_tree.iter_node_and_descendants(expn_id).flat_map(|n| n.spans.iter().map(|s| s.bcb));

    let bcb = match node.expn_kind {
        // For bang-macros (e.g. `assert!`, `trace!`) and for `await`, taking
        // the "first" BCB in dominator order seems to give good results.
        ExpnKind::Macro(MacroKind::Bang, _) | ExpnKind::Desugaring(DesugaringKind::Await) => {
            bcbs.min_by(|&a, &b| graph.cmp_in_dominator_order(a, b))?
        }
        // For other kinds of expansion, taking the "last" (most-dominated) BCB
        // seems to give good results.
        _ => bcbs.max_by(|&a, &b| graph.cmp_in_dominator_order(a, b))?,
    };

    // For bang-macro expansions, limit the call-site span to just the macro
    // name plus `!`, excluding the macro arguments.
    let mut span = node.call_site?;
    if matches!(node.expn_kind, ExpnKind::Macro(MacroKind::Bang, _)) {
        span = tcx.sess.source_map().span_through_char(span, '!');
    }

    Some(Covspan { span, bcb })
}

/// Discard all covspans that overlap a hole.
///
/// The lists of covspans and holes must be sorted, and any holes that overlap
/// with each other must have already been merged.
fn discard_spans_overlapping_holes(covspans: &mut Vec<Covspan>, holes: &[Hole]) {
    debug_assert!(covspans.is_sorted_by(|a, b| compare_spans(a.span, b.span).is_le()));
    debug_assert!(holes.is_sorted_by(|a, b| compare_spans(a.span, b.span).is_le()));
    debug_assert!(holes.array_windows().all(|[a, b]| !a.span.overlaps_or_adjacent(b.span)));

    let mut curr_hole = 0usize;
    let mut overlaps_hole = |covspan: &Covspan| -> bool {
        while let Some(hole) = holes.get(curr_hole) {
            // Both lists are sorted, so we can permanently skip any holes that
            // end before the start of the current span.
            if hole.span.hi() <= covspan.span.lo() {
                curr_hole += 1;
                continue;
            }

            return hole.span.overlaps(covspan.span);
        }

        // No holes left, so this covspan doesn't overlap with any holes.
        false
    };

    covspans.retain(|covspan| !overlaps_hole(covspan));
}

/// Takes a list of sorted spans extracted from MIR, and "refines"
/// those spans by removing spans that overlap in unwanted ways.
#[instrument(level = "debug")]
fn remove_unwanted_overlapping_spans(sorted_spans: Vec<Covspan>) -> Vec<Covspan> {
    debug_assert!(sorted_spans.is_sorted_by(|a, b| compare_spans(a.span, b.span).is_le()));

    // Holds spans that have been read from the input vector, but haven't yet
    // been committed to the output vector.
    let mut pending = vec![];
    let mut refined = vec![];

    for curr in sorted_spans {
        pending.retain(|prev: &Covspan| {
            if prev.span.hi() <= curr.span.lo() {
                // There's no overlap between the previous/current covspans,
                // so move the previous one into the refined list.
                refined.push(prev.clone());
                false
            } else {
                // Otherwise, retain the previous covspan only if it has the
                // same BCB. This tends to discard long outer spans that enclose
                // smaller inner spans with different control flow.
                prev.bcb == curr.bcb
            }
        });
        pending.push(curr);
    }

    // Drain the rest of the pending list into the refined list.
    refined.extend(pending);
    refined
}

#[derive(Clone, Debug)]
struct Covspan {
    span: Span,
    bcb: BasicCoverageBlock,
}

impl Covspan {
    /// If `self` and `other` can be merged, mutates `self.span` to also
    /// include `other.span` and returns true.
    ///
    /// Two covspans can be merged if they have the same BCB, and they are
    /// overlapping or adjacent.
    fn merge_if_eligible(&mut self, other: &Self) -> bool {
        let eligible_for_merge =
            |a: &Self, b: &Self| (a.bcb == b.bcb) && a.span.overlaps_or_adjacent(b.span);

        if eligible_for_merge(self, other) {
            self.span = self.span.to(other.span);
            true
        } else {
            false
        }
    }
}

/// Compares two spans in (lo ascending, hi descending) order.
fn compare_spans(a: Span, b: Span) -> std::cmp::Ordering {
    // First sort by span start.
    Ord::cmp(&a.lo(), &b.lo())
        // If span starts are the same, sort by span end in reverse order.
        // This ensures that if spans A and B are adjacent in the list,
        // and they overlap but are not equal, then either:
        // - Span A extends further left, or
        // - Both have the same start and span A extends further right
        .then_with(|| Ord::cmp(&a.hi(), &b.hi()).reverse())
}

fn ensure_non_empty_span(source_map: &SourceMap, span: Span) -> Option<Span> {
    if !span.is_empty() {
        return Some(span);
    }

    // The span is empty, so try to enlarge it to cover an adjacent '{' or '}'.
    source_map
        .span_to_source(span, |src, start, end| try {
            // Adjusting span endpoints by `BytePos(1)` is normally a bug,
            // but in this case we have specifically checked that the character
            // we're skipping over is one of two specific ASCII characters, so
            // adjusting by exactly 1 byte is correct.
            if src.as_bytes().get(end).copied() == Some(b'{') {
                Some(span.with_hi(span.hi() + BytePos(1)))
            } else if start > 0 && src.as_bytes()[start - 1] == b'}' {
                Some(span.with_lo(span.lo() - BytePos(1)))
            } else {
                None
            }
        })
        .ok()?
}
