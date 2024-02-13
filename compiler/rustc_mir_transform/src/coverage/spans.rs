use rustc_data_structures::graph::WithNumNodes;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir;
use rustc_span::{BytePos, Span};

use crate::coverage::graph::{BasicCoverageBlock, CoverageGraph, START_BCB};
use crate::coverage::spans::from_mir::SpanFromMir;
use crate::coverage::ExtractedHirInfo;

mod from_mir;

#[derive(Clone, Copy, Debug)]
pub(super) enum BcbMappingKind {
    /// Associates an ordinary executable code span with its corresponding BCB.
    Code(BasicCoverageBlock),
}

#[derive(Debug)]
pub(super) struct BcbMapping {
    pub(super) kind: BcbMappingKind,
    pub(super) span: Span,
}

pub(super) struct CoverageSpans {
    bcb_has_mappings: BitSet<BasicCoverageBlock>,
    mappings: Vec<BcbMapping>,
}

impl CoverageSpans {
    pub(super) fn bcb_has_coverage_spans(&self, bcb: BasicCoverageBlock) -> bool {
        self.bcb_has_mappings.contains(bcb)
    }

    pub(super) fn all_bcb_mappings(&self) -> impl Iterator<Item = &BcbMapping> {
        self.mappings.iter()
    }
}

/// Extracts coverage-relevant spans from MIR, and associates them with
/// their corresponding BCBs.
///
/// Returns `None` if no coverage-relevant spans could be extracted.
pub(super) fn generate_coverage_spans(
    mir_body: &mir::Body<'_>,
    hir_info: &ExtractedHirInfo,
    basic_coverage_blocks: &CoverageGraph,
) -> Option<CoverageSpans> {
    let mut mappings = vec![];

    if hir_info.is_async_fn {
        // An async function desugars into a function that returns a future,
        // with the user code wrapped in a closure. Any spans in the desugared
        // outer function will be unhelpful, so just keep the signature span
        // and ignore all of the spans in the MIR body.
        if let Some(span) = hir_info.fn_sig_span_extended {
            mappings.push(BcbMapping { kind: BcbMappingKind::Code(START_BCB), span });
        }
    } else {
        let sorted_spans = from_mir::mir_to_initial_sorted_coverage_spans(
            mir_body,
            hir_info,
            basic_coverage_blocks,
        );
        let coverage_spans = SpansRefiner::refine_sorted_spans(basic_coverage_blocks, sorted_spans);
        mappings.extend(coverage_spans.into_iter().map(|RefinedCovspan { bcb, span, .. }| {
            // Each span produced by the generator represents an ordinary code region.
            BcbMapping { kind: BcbMappingKind::Code(bcb), span }
        }));
    }

    if mappings.is_empty() {
        return None;
    }

    // Identify which BCBs have one or more mappings.
    let mut bcb_has_mappings = BitSet::new_empty(basic_coverage_blocks.num_nodes());
    let mut insert = |bcb| {
        bcb_has_mappings.insert(bcb);
    };
    for &BcbMapping { kind, span: _ } in &mappings {
        match kind {
            BcbMappingKind::Code(bcb) => insert(bcb),
        }
    }

    Some(CoverageSpans { bcb_has_mappings, mappings })
}

#[derive(Debug)]
struct CurrCovspan {
    /// This is used as the basis for [`PrevCovspan::original_span`], so it must
    /// not be modified.
    span: Span,
    bcb: BasicCoverageBlock,
    is_closure: bool,
}

impl CurrCovspan {
    fn new(span: Span, bcb: BasicCoverageBlock, is_closure: bool) -> Self {
        Self { span, bcb, is_closure }
    }

    fn into_prev(self) -> PrevCovspan {
        let Self { span, bcb, is_closure } = self;
        PrevCovspan { original_span: span, span, bcb, merged_spans: vec![span], is_closure }
    }

    fn into_refined(self) -> RefinedCovspan {
        // This is only called in cases where `curr` is a closure span that has
        // been carved out of `prev`.
        debug_assert!(self.is_closure);
        self.into_prev().into_refined()
    }
}

#[derive(Debug)]
struct PrevCovspan {
    original_span: Span,
    span: Span,
    bcb: BasicCoverageBlock,
    /// List of all the original spans from MIR that have been merged into this
    /// span. Mainly used to precisely skip over gaps when truncating a span.
    merged_spans: Vec<Span>,
    is_closure: bool,
}

impl PrevCovspan {
    fn is_mergeable(&self, other: &CurrCovspan) -> bool {
        self.bcb == other.bcb && !self.is_closure && !other.is_closure
    }

    fn merge_from(&mut self, other: &CurrCovspan) {
        debug_assert!(self.is_mergeable(other));
        self.span = self.span.to(other.span);
        self.merged_spans.push(other.span);
    }

    fn cutoff_statements_at(&mut self, cutoff_pos: BytePos) {
        self.merged_spans.retain(|span| span.hi() <= cutoff_pos);
        if let Some(max_hi) = self.merged_spans.iter().map(|span| span.hi()).max() {
            self.span = self.span.with_hi(max_hi);
        }
    }

    fn into_dup(self) -> DuplicateCovspan {
        let Self { original_span, span, bcb, merged_spans: _, is_closure } = self;
        // Only unmodified spans end up in `pending_dups`.
        debug_assert_eq!(original_span, span);
        DuplicateCovspan { span, bcb, is_closure }
    }

    fn refined_copy(&self) -> RefinedCovspan {
        let &Self { original_span: _, span, bcb, merged_spans: _, is_closure } = self;
        RefinedCovspan { span, bcb, is_closure }
    }

    fn into_refined(self) -> RefinedCovspan {
        self.refined_copy()
    }
}

#[derive(Debug)]
struct DuplicateCovspan {
    span: Span,
    bcb: BasicCoverageBlock,
    is_closure: bool,
}

impl DuplicateCovspan {
    /// Returns a copy of this covspan, as a [`RefinedCovspan`].
    /// Should only be called in places that would otherwise clone this covspan.
    fn refined_copy(&self) -> RefinedCovspan {
        let &Self { span, bcb, is_closure } = self;
        RefinedCovspan { span, bcb, is_closure }
    }

    fn into_refined(self) -> RefinedCovspan {
        // Even though we consume self, we can just reuse the copying impl.
        self.refined_copy()
    }
}

#[derive(Debug)]
struct RefinedCovspan {
    span: Span,
    bcb: BasicCoverageBlock,
    is_closure: bool,
}

impl RefinedCovspan {
    fn is_mergeable(&self, other: &Self) -> bool {
        self.bcb == other.bcb && !self.is_closure && !other.is_closure
    }

    fn merge_from(&mut self, other: &Self) {
        debug_assert!(self.is_mergeable(other));
        self.span = self.span.to(other.span);
    }
}

/// Converts the initial set of coverage spans (one per MIR `Statement` or `Terminator`) into a
/// minimal set of coverage spans, using the BCB CFG to determine where it is safe and useful to:
///
///  * Remove duplicate source code coverage regions
///  * Merge spans that represent continuous (both in source code and control flow), non-branching
///    execution
///  * Carve out (leave uncovered) any span that will be counted by another MIR (notably, closures)
struct SpansRefiner<'a> {
    /// The BasicCoverageBlock Control Flow Graph (BCB CFG).
    basic_coverage_blocks: &'a CoverageGraph,

    /// The initial set of coverage spans, sorted by `Span` (`lo` and `hi`) and by relative
    /// dominance between the `BasicCoverageBlock`s of equal `Span`s.
    sorted_spans_iter: std::vec::IntoIter<SpanFromMir>,

    /// The current coverage span to compare to its `prev`, to possibly merge, discard, force the
    /// discard of the `prev` (and or `pending_dups`), or keep both (with `prev` moved to
    /// `pending_dups`). If `curr` is not discarded or merged, it becomes `prev` for the next
    /// iteration.
    some_curr: Option<CurrCovspan>,

    /// The coverage span from a prior iteration; typically assigned from that iteration's `curr`.
    /// If that `curr` was discarded, `prev` retains its value from the previous iteration.
    some_prev: Option<PrevCovspan>,

    /// One or more coverage spans with the same `Span` but different `BasicCoverageBlock`s, and
    /// no `BasicCoverageBlock` in this list dominates another `BasicCoverageBlock` in the list.
    /// If a new `curr` span also fits this criteria (compared to an existing list of
    /// `pending_dups`), that `curr` moves to `prev` before possibly being added to
    /// the `pending_dups` list, on the next iteration. As a result, if `prev` and `pending_dups`
    /// have the same `Span`, the criteria for `pending_dups` holds for `prev` as well: a `prev`
    /// with a matching `Span` does not dominate any `pending_dup` and no `pending_dup` dominates a
    /// `prev` with a matching `Span`)
    pending_dups: Vec<DuplicateCovspan>,

    /// The final coverage spans to add to the coverage map. A `Counter` or `Expression`
    /// will also be injected into the MIR for each BCB that has associated spans.
    refined_spans: Vec<RefinedCovspan>,
}

impl<'a> SpansRefiner<'a> {
    /// Takes the initial list of (sorted) spans extracted from MIR, and "refines"
    /// them by merging compatible adjacent spans, removing redundant spans,
    /// and carving holes in spans when they overlap in unwanted ways.
    fn refine_sorted_spans(
        basic_coverage_blocks: &'a CoverageGraph,
        sorted_spans: Vec<SpanFromMir>,
    ) -> Vec<RefinedCovspan> {
        let this = Self {
            basic_coverage_blocks,
            sorted_spans_iter: sorted_spans.into_iter(),
            some_curr: None,
            some_prev: None,
            pending_dups: Vec::new(),
            refined_spans: Vec::with_capacity(basic_coverage_blocks.num_nodes() * 2),
        };

        this.to_refined_spans()
    }

    /// Iterate through the sorted coverage spans, and return the refined list of merged and
    /// de-duplicated spans.
    fn to_refined_spans(mut self) -> Vec<RefinedCovspan> {
        while self.next_coverage_span() {
            // For the first span we don't have `prev` set, so most of the
            // span-processing steps don't make sense yet.
            if self.some_prev.is_none() {
                debug!("  initial span");
                continue;
            }

            // The remaining cases assume that `prev` and `curr` are set.
            let prev = self.prev();
            let curr = self.curr();

            if prev.is_mergeable(curr) {
                debug!("  same bcb (and neither is a closure), merge with prev={prev:?}");
                let curr = self.take_curr();
                self.prev_mut().merge_from(&curr);
            } else if prev.span.hi() <= curr.span.lo() {
                debug!(
                    "  different bcbs and disjoint spans, so keep curr for next iter, and add prev={prev:?}",
                );
                let prev = self.take_prev().into_refined();
                self.refined_spans.push(prev);
            } else if prev.is_closure {
                // drop any equal or overlapping span (`curr`) and keep `prev` to test again in the
                // next iter
                debug!(
                    "  curr overlaps a closure (prev). Drop curr and keep prev for next iter. prev={prev:?}",
                );
                self.take_curr(); // Discards curr.
            } else if curr.is_closure {
                self.carve_out_span_for_closure();
            } else if prev.original_span == prev.span && prev.span == curr.span {
                // Prev and curr have the same span, and prev's span hasn't
                // been modified by other spans.
                self.update_pending_dups();
            } else {
                self.cutoff_prev_at_overlapping_curr();
            }
        }

        // Drain any remaining dups into the output.
        for dup in self.pending_dups.drain(..) {
            debug!("    ...adding at least one pending dup={:?}", dup);
            self.refined_spans.push(dup.into_refined());
        }

        // There is usually a final span remaining in `prev` after the loop ends,
        // so add it to the output as well.
        if let Some(prev) = self.some_prev.take() {
            debug!("    AT END, adding last prev={prev:?}");
            self.refined_spans.push(prev.into_refined());
        }

        // Do one last merge pass, to simplify the output.
        self.refined_spans.dedup_by(|b, a| {
            if a.is_mergeable(b) {
                debug!(?a, ?b, "merging list-adjacent refined spans");
                a.merge_from(b);
                true
            } else {
                false
            }
        });

        // Remove spans derived from closures, originally added to ensure the coverage
        // regions for the current function leave room for the closure's own coverage regions
        // (injected separately, from the closure's own MIR).
        self.refined_spans.retain(|covspan| !covspan.is_closure);
        self.refined_spans
    }

    #[track_caller]
    fn curr(&self) -> &CurrCovspan {
        self.some_curr.as_ref().unwrap_or_else(|| bug!("some_curr is None (curr)"))
    }

    /// If called, then the next call to `next_coverage_span()` will *not* update `prev` with the
    /// `curr` coverage span.
    #[track_caller]
    fn take_curr(&mut self) -> CurrCovspan {
        self.some_curr.take().unwrap_or_else(|| bug!("some_curr is None (take_curr)"))
    }

    #[track_caller]
    fn prev(&self) -> &PrevCovspan {
        self.some_prev.as_ref().unwrap_or_else(|| bug!("some_prev is None (prev)"))
    }

    #[track_caller]
    fn prev_mut(&mut self) -> &mut PrevCovspan {
        self.some_prev.as_mut().unwrap_or_else(|| bug!("some_prev is None (prev_mut)"))
    }

    #[track_caller]
    fn take_prev(&mut self) -> PrevCovspan {
        self.some_prev.take().unwrap_or_else(|| bug!("some_prev is None (take_prev)"))
    }

    /// If there are `pending_dups` but `prev` is not a matching dup (`prev.span` doesn't match the
    /// `pending_dups` spans), then one of the following two things happened during the previous
    /// iteration:
    ///   * the previous `curr` span (which is now `prev`) was not a duplicate of the pending_dups
    ///     (in which case there should be at least two spans in `pending_dups`); or
    ///   * the `span` of `prev` was modified by `curr_mut().merge_from(prev)` (in which case
    ///     `pending_dups` could have as few as one span)
    /// In either case, no more spans will match the span of `pending_dups`, so
    /// add the `pending_dups` if they don't overlap `curr`, and clear the list.
    fn maybe_flush_pending_dups(&mut self) {
        let Some(last_dup) = self.pending_dups.last() else { return };
        if last_dup.span == self.prev().span {
            return;
        }

        debug!(
            "    SAME spans, but pending_dups are NOT THE SAME, so BCBs matched on \
            previous iteration, or prev started a new disjoint span"
        );
        if last_dup.span.hi() <= self.curr().span.lo() {
            for dup in self.pending_dups.drain(..) {
                debug!("    ...adding at least one pending={:?}", dup);
                self.refined_spans.push(dup.into_refined());
            }
        } else {
            self.pending_dups.clear();
        }
        assert!(self.pending_dups.is_empty());
    }

    /// Advance `prev` to `curr` (if any), and `curr` to the next coverage span in sorted order.
    fn next_coverage_span(&mut self) -> bool {
        if let Some(curr) = self.some_curr.take() {
            self.some_prev = Some(curr.into_prev());
        }
        while let Some(curr) = self.sorted_spans_iter.next() {
            debug!("FOR curr={:?}", curr);
            if let Some(prev) = &self.some_prev
                && prev.span.lo() > curr.span.lo()
            {
                // Skip curr because prev has already advanced beyond the end of curr.
                // This can only happen if a prior iteration updated `prev` to skip past
                // a region of code, such as skipping past a closure.
                debug!(
                    "  prev.span starts after curr.span, so curr will be dropped (skipping past \
                    closure?); prev={prev:?}",
                );
            } else {
                self.some_curr = Some(CurrCovspan::new(curr.span, curr.bcb, curr.is_closure));
                self.maybe_flush_pending_dups();
                return true;
            }
        }
        false
    }

    /// If `prev`s span extends left of the closure (`curr`), carve out the closure's span from
    /// `prev`'s span. (The closure's coverage counters will be injected when processing the
    /// closure's own MIR.) Add the portion of the span to the left of the closure; and if the span
    /// extends to the right of the closure, update `prev` to that portion of the span. For any
    /// `pending_dups`, repeat the same process.
    fn carve_out_span_for_closure(&mut self) {
        let prev = self.prev();
        let curr = self.curr();

        let left_cutoff = curr.span.lo();
        let right_cutoff = curr.span.hi();
        let has_pre_closure_span = prev.span.lo() < right_cutoff;
        let has_post_closure_span = prev.span.hi() > right_cutoff;

        if has_pre_closure_span {
            let mut pre_closure = self.prev().refined_copy();
            pre_closure.span = pre_closure.span.with_hi(left_cutoff);
            debug!("  prev overlaps a closure. Adding span for pre_closure={:?}", pre_closure);

            for mut dup in self.pending_dups.iter().map(DuplicateCovspan::refined_copy) {
                dup.span = dup.span.with_hi(left_cutoff);
                debug!("    ...and at least one pre_closure dup={:?}", dup);
                self.refined_spans.push(dup);
            }

            self.refined_spans.push(pre_closure);
        }

        if has_post_closure_span {
            // Mutate `prev.span` to start after the closure (and discard curr).
            self.prev_mut().span = self.prev().span.with_lo(right_cutoff);
            debug!("  Mutated prev.span to start after the closure. prev={:?}", self.prev());

            for dup in &mut self.pending_dups {
                debug!("    ...and at least one overlapping dup={:?}", dup);
                dup.span = dup.span.with_lo(right_cutoff);
            }

            // Prevent this curr from becoming prev.
            let closure_covspan = self.take_curr().into_refined();
            self.refined_spans.push(closure_covspan); // since self.prev() was already updated
        } else {
            self.pending_dups.clear();
        }
    }

    /// Called if `curr.span` equals `prev.original_span` (and potentially equal to all
    /// `pending_dups` spans, if any). Keep in mind, `prev.span()` may have been changed.
    /// If prev.span() was merged into other spans (with matching BCB, for instance),
    /// `prev.span.hi()` will be greater than (further right of) `prev.original_span.hi()`.
    /// If prev.span() was split off to the right of a closure, prev.span().lo() will be
    /// greater than prev.original_span.lo(). The actual span of `prev.original_span` is
    /// not as important as knowing that `prev()` **used to have the same span** as `curr()`,
    /// which means their sort order is still meaningful for determining the dominator
    /// relationship.
    ///
    /// When two coverage spans have the same `Span`, dominated spans can be discarded; but if
    /// neither coverage span dominates the other, both (or possibly more than two) are held,
    /// until their disposition is determined. In this latter case, the `prev` dup is moved into
    /// `pending_dups` so the new `curr` dup can be moved to `prev` for the next iteration.
    fn update_pending_dups(&mut self) {
        let prev_bcb = self.prev().bcb;
        let curr_bcb = self.curr().bcb;

        // Equal coverage spans are ordered by dominators before dominated (if any), so it should be
        // impossible for `curr` to dominate any previous coverage span.
        debug_assert!(!self.basic_coverage_blocks.dominates(curr_bcb, prev_bcb));

        // `prev` is a duplicate of `curr`, so add it to the list of pending dups.
        // If it dominates `curr`, it will be removed by the subsequent discard step.
        let prev = self.take_prev().into_dup();
        debug!(?prev, "adding prev to pending dups");
        self.pending_dups.push(prev);

        let initial_pending_count = self.pending_dups.len();
        if initial_pending_count > 0 {
            self.pending_dups
                .retain(|dup| !self.basic_coverage_blocks.dominates(dup.bcb, curr_bcb));

            let n_discarded = initial_pending_count - self.pending_dups.len();
            if n_discarded > 0 {
                debug!(
                    "  discarded {n_discarded} of {initial_pending_count} pending_dups that dominated curr",
                );
            }
        }
    }

    /// `curr` overlaps `prev`. If `prev`s span extends left of `curr`s span, keep _only_
    /// statements that end before `curr.lo()` (if any), and add the portion of the
    /// combined span for those statements. Any other statements have overlapping spans
    /// that can be ignored because `curr` and/or other upcoming statements/spans inside
    /// the overlap area will produce their own counters. This disambiguation process
    /// avoids injecting multiple counters for overlapping spans, and the potential for
    /// double-counting.
    fn cutoff_prev_at_overlapping_curr(&mut self) {
        debug!(
            "  different bcbs, overlapping spans, so ignore/drop pending and only add prev \
            if it has statements that end before curr; prev={:?}",
            self.prev()
        );
        if self.pending_dups.is_empty() {
            let curr_span = self.curr().span;
            self.prev_mut().cutoff_statements_at(curr_span.lo());
            if self.prev().merged_spans.is_empty() {
                debug!("  ... no non-overlapping statements to add");
            } else {
                debug!("  ... adding modified prev={:?}", self.prev());
                let prev = self.take_prev().into_refined();
                self.refined_spans.push(prev);
            }
        } else {
            // with `pending_dups`, `prev` cannot have any statements that don't overlap
            self.pending_dups.clear();
        }
    }
}
