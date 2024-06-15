use std::collections::VecDeque;

use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir;
use rustc_span::Span;

use crate::coverage::graph::{BasicCoverageBlock, CoverageGraph};
use crate::coverage::mappings;
use crate::coverage::spans::from_mir::{
    extract_covspans_and_holes_from_mir, ExtractedCovspans, SpanFromMir,
};
use crate::coverage::ExtractedHirInfo;

mod from_mir;

// FIXME(#124545) It's awkward that we have to re-export this, because it's an
// internal detail of `from_mir` that is also needed when handling branch and
// MC/DC spans. Ideally we would find a more natural home for it.
pub(super) use from_mir::unexpand_into_body_span_with_visible_macro;

pub(super) fn extract_refined_covspans(
    mir_body: &mir::Body<'_>,
    hir_info: &ExtractedHirInfo,
    basic_coverage_blocks: &CoverageGraph,
    code_mappings: &mut impl Extend<mappings::CodeMapping>,
) {
    let ExtractedCovspans { mut covspans, mut holes } =
        extract_covspans_and_holes_from_mir(mir_body, hir_info, basic_coverage_blocks);

    covspans.sort_by(|a, b| basic_coverage_blocks.cmp_in_dominator_order(a.bcb, b.bcb));
    remove_unwanted_macro_spans(&mut covspans);
    split_visible_macro_spans(&mut covspans);

    let compare_covspans = |a: &SpanFromMir, b: &SpanFromMir| {
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
    holes.sort_by(|a, b| compare_spans(a.span, b.span));
    holes.dedup_by(|b, a| a.merge_if_overlapping_or_adjacent(b));

    // Now we're ready to start carving holes out of the initial coverage spans,
    // and grouping them in buckets separated by the holes.

    let mut input_covspans = VecDeque::from(covspans);
    let mut fragments: Vec<SpanFromMir> = vec![];

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

    for mut covspans in buckets {
        // Make sure each individual bucket is internally sorted.
        covspans.sort_by(compare_covspans);

        let covspans = refine_sorted_spans(covspans);
        code_mappings.extend(covspans.into_iter().map(|RefinedCovspan { span, bcb }| {
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
/// (The input spans should be sorted in BCB dominator order, so that the
/// retained "first" span is likely to dominate the others.)
fn remove_unwanted_macro_spans(covspans: &mut Vec<SpanFromMir>) {
    let mut seen_macro_spans = FxHashSet::default();
    covspans.retain(|covspan| {
        // Ignore (retain) non-macro-expansion spans.
        if covspan.visible_macro.is_none() {
            return true;
        }

        // Retain only the first macro-expanded covspan with this span.
        seen_macro_spans.insert(covspan.span)
    });
}

/// When a span corresponds to a macro invocation that is visible from the
/// function body, split it into two parts. The first part covers just the
/// macro name plus `!`, and the second part covers the rest of the macro
/// invocation. This seems to give better results for code that uses macros.
fn split_visible_macro_spans(covspans: &mut Vec<SpanFromMir>) {
    let mut extra_spans = vec![];

    covspans.retain(|covspan| {
        let Some(visible_macro) = covspan.visible_macro else { return true };

        let split_len = visible_macro.as_str().len() as u32 + 1;
        let (before, after) = covspan.span.split_at(split_len);
        if !covspan.span.contains(before) || !covspan.span.contains(after) {
            // Something is unexpectedly wrong with the split point.
            // The debug assertion in `split_at` will have already caught this,
            // but in release builds it's safer to do nothing and maybe get a
            // bug report for unexpected coverage, rather than risk an ICE.
            return true;
        }

        extra_spans.push(SpanFromMir::new(before, covspan.visible_macro, covspan.bcb));
        extra_spans.push(SpanFromMir::new(after, covspan.visible_macro, covspan.bcb));
        false // Discard the original covspan that we just split.
    });

    // The newly-split spans are added at the end, so any previous sorting
    // is not preserved.
    covspans.extend(extra_spans);
}

#[derive(Debug)]
struct RefinedCovspan {
    span: Span,
    bcb: BasicCoverageBlock,
}

impl RefinedCovspan {
    fn is_mergeable(&self, other: &Self) -> bool {
        self.bcb == other.bcb
    }

    fn merge_from(&mut self, other: &Self) {
        debug_assert!(self.is_mergeable(other));
        self.span = self.span.to(other.span);
    }
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
/// those spans by removing spans that overlap in unwanted ways, and by merging
/// compatible adjacent spans.
#[instrument(level = "debug")]
fn refine_sorted_spans(sorted_spans: Vec<SpanFromMir>) -> Vec<RefinedCovspan> {
    // Holds spans that have been read from the input vector, but haven't yet
    // been committed to the output vector.
    let mut pending = vec![];
    let mut refined = vec![];

    for curr in sorted_spans {
        pending.retain(|prev: &SpanFromMir| {
            if prev.span.hi() <= curr.span.lo() {
                // There's no overlap between the previous/current covspans,
                // so move the previous one into the refined list.
                refined.push(RefinedCovspan { span: prev.span, bcb: prev.bcb });
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
    for prev in pending {
        refined.push(RefinedCovspan { span: prev.span, bcb: prev.bcb });
    }

    // Do one last merge pass, to simplify the output.
    debug!(?refined, "before merge");
    refined.dedup_by(|b, a| {
        if a.is_mergeable(b) {
            debug!(?a, ?b, "merging list-adjacent refined spans");
            a.merge_from(b);
            true
        } else {
            false
        }
    });
    debug!(?refined, "after merge");

    refined
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
