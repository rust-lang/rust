use std::collections::VecDeque;

use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir;
use rustc_span::{DesugaringKind, ExpnKind, MacroKind, Span};
use tracing::{debug, debug_span, instrument};

use crate::coverage::graph::{BasicCoverageBlock, CoverageGraph};
use crate::coverage::spans::from_mir::{
    ExtractedCovspans, Hole, SpanFromMir, extract_covspans_from_mir,
};
use crate::coverage::{ExtractedHirInfo, mappings};

mod from_mir;

pub(super) fn extract_refined_covspans(
    mir_body: &mir::Body<'_>,
    hir_info: &ExtractedHirInfo,
    basic_coverage_blocks: &CoverageGraph,
    code_mappings: &mut impl Extend<mappings::CodeMapping>,
) {
    let ExtractedCovspans { mut covspans } =
        extract_covspans_from_mir(mir_body, hir_info, basic_coverage_blocks);

    // First, perform the passes that need macro information.
    covspans.sort_by(|a, b| basic_coverage_blocks.cmp_in_dominator_order(a.bcb, b.bcb));
    remove_unwanted_expansion_spans(&mut covspans);
    split_visible_macro_spans(&mut covspans);

    // We no longer need the extra information in `SpanFromMir`, so convert to `Covspan`.
    let mut covspans = covspans.into_iter().map(SpanFromMir::into_covspan).collect::<Vec<_>>();

    let compare_covspans = |a: &Covspan, b: &Covspan| {
        compare_spans(a.span, b.span)
            // After deduplication, we want to keep only the most-dominated BCB.
            .then_with(|| basic_coverage_blocks.cmp_in_dominator_order(a.bcb, b.bcb).reverse())
    };
    covspans.sort_by(compare_covspans);

    // Among covspans with the same span, keep only one,
    // preferring the one with the most-dominated BCB.
    // (Ideally we should try to preserve _all_ non-dominating BCBs, but that
    // requires a lot more complexity in the span refiner, for little benefit.)
    covspans.dedup_by(|b, a| a.span.source_equal(b.span));

    // Sort the holes, and merge overlapping/adjacent holes.
    let mut holes = hir_info.hole_spans.iter().map(|&span| Hole { span }).collect::<Vec<_>>();
    holes.sort_by(|a, b| compare_spans(a.span, b.span));
    holes.dedup_by(|b, a| a.merge_if_overlapping_or_adjacent(b));

    // Split the covspans into separate buckets that don't overlap any holes.
    let buckets = divide_spans_into_buckets(covspans, &holes);

    for mut covspans in buckets {
        // Make sure each individual bucket is internally sorted.
        covspans.sort_by(compare_covspans);
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
/// function body, split it into two parts. The first part covers just the
/// macro name plus `!`, and the second part covers the rest of the macro
/// invocation. This seems to give better results for code that uses macros.
fn split_visible_macro_spans(covspans: &mut Vec<SpanFromMir>) {
    let mut extra_spans = vec![];

    covspans.retain(|covspan| {
        let Some(ExpnKind::Macro(MacroKind::Bang, visible_macro)) = covspan.expn_kind else {
            return true;
        };

        let split_len = visible_macro.as_str().len() as u32 + 1;
        let (before, after) = covspan.span.split_at(split_len);
        if !covspan.span.contains(before) || !covspan.span.contains(after) {
            // Something is unexpectedly wrong with the split point.
            // The debug assertion in `split_at` will have already caught this,
            // but in release builds it's safer to do nothing and maybe get a
            // bug report for unexpected coverage, rather than risk an ICE.
            return true;
        }

        extra_spans.push(SpanFromMir::new(before, covspan.expn_kind.clone(), covspan.bcb));
        extra_spans.push(SpanFromMir::new(after, covspan.expn_kind.clone(), covspan.bcb));
        false // Discard the original covspan that we just split.
    });

    // The newly-split spans are added at the end, so any previous sorting
    // is not preserved.
    covspans.extend(extra_spans);
}

/// Uses the holes to divide the given covspans into buckets, such that:
/// - No span in any hole overlaps a bucket (truncating the spans if necessary).
/// - The spans in each bucket are strictly after all spans in previous buckets,
///   and strictly before all spans in subsequent buckets.
///
/// The resulting buckets are sorted relative to each other, but might not be
/// internally sorted.
#[instrument(level = "debug")]
fn divide_spans_into_buckets(input_covspans: Vec<Covspan>, holes: &[Hole]) -> Vec<Vec<Covspan>> {
    debug_assert!(input_covspans.is_sorted_by(|a, b| compare_spans(a.span, b.span).is_le()));
    debug_assert!(holes.is_sorted_by(|a, b| compare_spans(a.span, b.span).is_le()));

    // Now we're ready to start carving holes out of the initial coverage spans,
    // and grouping them in buckets separated by the holes.

    let mut input_covspans = VecDeque::from(input_covspans);
    let mut fragments = vec![];

    // For each hole:
    // - Identify the spans that are entirely or partly before the hole.
    // - Put those spans in a corresponding bucket, truncated to the start of the hole.
    // - If one of those spans also extends after the hole, put the rest of it
    //   in a "fragments" vector that is processed by the next hole.
    let mut buckets = (0..holes.len()).map(|_| vec![]).collect::<Vec<_>>();
    for (hole, bucket) in holes.iter().zip(&mut buckets) {
        let fragments_from_prev = std::mem::take(&mut fragments);

        // Only inspect spans that precede or overlap this hole,
        // leaving the rest to be inspected by later holes.
        // (This relies on the spans and holes both being sorted.)
        let relevant_input_covspans =
            drain_front_while(&mut input_covspans, |c| c.span.lo() < hole.span.hi());

        for covspan in fragments_from_prev.into_iter().chain(relevant_input_covspans) {
            let (before, after) = covspan.split_around_hole_span(hole.span);
            bucket.extend(before);
            fragments.extend(after);
        }
    }

    // After finding the spans before each hole, any remaining fragments/spans
    // form their own final bucket, after the final hole.
    // (If there were no holes, this will just be all of the initial spans.)
    fragments.extend(input_covspans);
    buckets.push(fragments);

    buckets
}

/// Similar to `.drain(..)`, but stops just before it would remove an item not
/// satisfying the predicate.
fn drain_front_while<'a, T>(
    queue: &'a mut VecDeque<T>,
    mut pred_fn: impl FnMut(&T) -> bool,
) -> impl Iterator<Item = T> + Captures<'a> {
    std::iter::from_fn(move || if pred_fn(queue.front()?) { queue.pop_front() } else { None })
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
    /// Splits this covspan into 0-2 parts:
    /// - The part that is strictly before the hole span, if any.
    /// - The part that is strictly after the hole span, if any.
    fn split_around_hole_span(&self, hole_span: Span) -> (Option<Self>, Option<Self>) {
        let before = try {
            let span = self.span.trim_end(hole_span)?;
            Self { span, ..*self }
        };
        let after = try {
            let span = self.span.trim_start(hole_span)?;
            Self { span, ..*self }
        };

        (before, after)
    }

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
