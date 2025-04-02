use std::collections::VecDeque;
use std::iter;

use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir;
use rustc_middle::ty::TyCtxt;
use rustc_span::{DesugaringKind, ExpnKind, MacroKind, Span};
use tracing::{debug, debug_span, instrument};

use crate::coverage::graph::{BasicCoverageBlock, CoverageGraph};
use crate::coverage::spans::from_mir::{Hole, RawSpanFromMir, SpanFromMir};
use crate::coverage::{ExtractedHirInfo, mappings, unexpand};

mod from_mir;

pub(super) fn extract_refined_covspans<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    hir_info: &ExtractedHirInfo,
    graph: &CoverageGraph,
    code_mappings: &mut impl Extend<mappings::CodeMapping>,
) {
    let &ExtractedHirInfo { body_span, .. } = hir_info;

    let raw_spans = from_mir::extract_raw_spans_from_mir(mir_body, graph);
    let mut covspans = raw_spans
        .into_iter()
        .filter_map(|RawSpanFromMir { raw_span, bcb }| try {
            let (span, expn_kind) =
                unexpand::unexpand_into_body_span_with_expn_kind(raw_span, body_span)?;
            // Discard any spans that fill the entire body, because they tend
            // to represent compiler-inserted code, e.g. implicitly returning `()`.
            if span.source_equal(body_span) {
                return None;
            };
            SpanFromMir { span, expn_kind, bcb }
        })
        .collect::<Vec<_>>();

    // Only proceed if we found at least one usable span.
    if covspans.is_empty() {
        return;
    }

    // Also add the adjusted function signature span, if available.
    // Otherwise, add a fake span at the start of the body, to avoid an ugly
    // gap between the start of the body and the first real span.
    // FIXME: Find a more principled way to solve this problem.
    covspans.push(SpanFromMir::for_fn_sig(
        hir_info.fn_sig_span_extended.unwrap_or_else(|| body_span.shrink_to_lo()),
    ));

    // First, perform the passes that need macro information.
    covspans.sort_by(|a, b| graph.cmp_in_dominator_order(a.bcb, b.bcb));
    remove_unwanted_expansion_spans(&mut covspans);
    shrink_visible_macro_spans(tcx, &mut covspans);

    // We no longer need the extra information in `SpanFromMir`, so convert to `Covspan`.
    let mut covspans = covspans.into_iter().map(SpanFromMir::into_covspan).collect::<Vec<_>>();

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

    // Split the covspans into separate buckets that don't overlap any holes.
    let buckets = divide_spans_into_buckets(covspans, &holes);

    for covspans in buckets {
        let _span = debug_span!("processing bucket", ?covspans).entered();

        let mut covspans = remove_unwanted_overlapping_spans(covspans);
        debug!(?covspans, "after removing overlaps");

        // Do one last merge pass, to simplify the output.
        covspans.dedup_by(|b, a| a.merge_if_eligible(b));
        debug!(?covspans, "after merge");

        code_mappings.extend(covspans.into_iter().map(|Covspan { span, bcb }| {
            // Each span produced by the refiner represents an ordinary code region.
            mappings::CodeMapping { span, bcb }
        }));
    }
}

/// Macros that expand into branches (e.g. `assert!`, `trace!`) tend to generate
/// multiple condition/consequent blocks that have the span of the whole macro
/// invocation, which is unhelpful. Keeping only the first such span seems to
/// give better mappings, so remove the others.
///
/// Similarly, `await` expands to a branch on the discriminant of `Poll`, which
/// leads to incorrect coverage if the `Future` is immediately ready (#98712).
///
/// (The input spans should be sorted in BCB dominator order, so that the
/// retained "first" span is likely to dominate the others.)
fn remove_unwanted_expansion_spans(covspans: &mut Vec<SpanFromMir>) {
    let mut deduplicated_spans = FxHashSet::default();

    covspans.retain(|covspan| {
        match covspan.expn_kind {
            // Retain only the first await-related or macro-expanded covspan with this span.
            Some(ExpnKind::Desugaring(DesugaringKind::Await)) => {
                deduplicated_spans.insert(covspan.span)
            }
            Some(ExpnKind::Macro(MacroKind::Bang, _)) => deduplicated_spans.insert(covspan.span),
            // Ignore (retain) other spans.
            _ => true,
        }
    });
}

/// When a span corresponds to a macro invocation that is visible from the
/// function body, truncate it to just the macro name plus `!`.
/// This seems to give better results for code that uses macros.
fn shrink_visible_macro_spans(tcx: TyCtxt<'_>, covspans: &mut Vec<SpanFromMir>) {
    let source_map = tcx.sess.source_map();

    for covspan in covspans {
        if matches!(covspan.expn_kind, Some(ExpnKind::Macro(MacroKind::Bang, _))) {
            covspan.span = source_map.span_through_char(covspan.span, '!');
        }
    }
}

/// Uses the holes to divide the given covspans into buckets, such that:
/// - No span in any hole overlaps a bucket (discarding spans if necessary).
/// - The spans in each bucket are strictly after all spans in previous buckets,
///   and strictly before all spans in subsequent buckets.
///
/// The lists of covspans and holes must be sorted.
/// The resulting buckets are sorted relative to each other, and each bucket's
/// contents are sorted.
#[instrument(level = "debug")]
fn divide_spans_into_buckets(input_covspans: Vec<Covspan>, holes: &[Hole]) -> Vec<Vec<Covspan>> {
    debug_assert!(input_covspans.is_sorted_by(|a, b| compare_spans(a.span, b.span).is_le()));
    debug_assert!(holes.is_sorted_by(|a, b| compare_spans(a.span, b.span).is_le()));

    // Now we're ready to start grouping spans into buckets separated by holes.

    let mut input_covspans = VecDeque::from(input_covspans);

    // For each hole:
    // - Identify the spans that are entirely or partly before the hole.
    // - Discard any that overlap with the hole.
    // - Add the remaining identified spans to the corresponding bucket.
    let mut buckets = (0..holes.len()).map(|_| vec![]).collect::<Vec<_>>();
    for (hole, bucket) in holes.iter().zip(&mut buckets) {
        bucket.extend(
            drain_front_while(&mut input_covspans, |c| c.span.lo() < hole.span.hi())
                .filter(|c| !c.span.overlaps(hole.span)),
        );
    }

    // Any remaining spans form their own final bucket, after the final hole.
    // (If there were no holes, this will just be all of the initial spans.)
    buckets.push(Vec::from(input_covspans));

    buckets
}

/// Similar to `.drain(..)`, but stops just before it would remove an item not
/// satisfying the predicate.
fn drain_front_while<'a, T>(
    queue: &'a mut VecDeque<T>,
    mut pred_fn: impl FnMut(&T) -> bool,
) -> impl Iterator<Item = T> {
    iter::from_fn(move || queue.pop_front_if(|x| pred_fn(x)))
}

/// Takes one of the buckets of (sorted) spans extracted from MIR, and "refines"
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
    /// If `self` and `other` can be merged (i.e. they have the same BCB),
    /// mutates `self.span` to also include `other.span` and returns true.
    ///
    /// Note that compatible covspans can be merged even if their underlying
    /// spans are not overlapping/adjacent; any space between them will also be
    /// part of the merged covspan.
    fn merge_if_eligible(&mut self, other: &Self) -> bool {
        if self.bcb != other.bcb {
            return false;
        }

        self.span = self.span.to(other.span);
        true
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
