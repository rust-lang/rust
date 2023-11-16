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
    /// Associates a branch span with BCBs for its true and false arms.
    Branch { true_bcb: BasicCoverageBlock, false_bcb: BasicCoverageBlock },
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
        let coverage_spans = SpansRefiner::refine_sorted_spans(sorted_spans);
        mappings.extend(coverage_spans.into_iter().map(|RefinedCovspan { bcb, span, .. }| {
            // Each span produced by the generator represents an ordinary code region.
            BcbMapping { kind: BcbMappingKind::Code(bcb), span }
        }));

        mappings.extend(from_mir::extract_branch_mappings(
            mir_body,
            hir_info.body_span,
            basic_coverage_blocks,
        ));
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
            BcbMappingKind::Branch { true_bcb, false_bcb } => {
                insert(true_bcb);
                insert(false_bcb);
            }
        }
    }

    Some(CoverageSpans { bcb_has_mappings, mappings })
}

#[derive(Debug)]
struct CurrCovspan {
    span: Span,
    bcb: BasicCoverageBlock,
    is_hole: bool,
}

impl CurrCovspan {
    fn new(span: Span, bcb: BasicCoverageBlock, is_hole: bool) -> Self {
        Self { span, bcb, is_hole }
    }

    fn into_prev(self) -> PrevCovspan {
        let Self { span, bcb, is_hole } = self;
        PrevCovspan { span, bcb, merged_spans: vec![span], is_hole }
    }

    fn into_refined(self) -> RefinedCovspan {
        // This is only called in cases where `curr` is a hole span that has
        // been carved out of `prev`.
        debug_assert!(self.is_hole);
        self.into_prev().into_refined()
    }
}

#[derive(Debug)]
struct PrevCovspan {
    span: Span,
    bcb: BasicCoverageBlock,
    /// List of all the original spans from MIR that have been merged into this
    /// span. Mainly used to precisely skip over gaps when truncating a span.
    merged_spans: Vec<Span>,
    is_hole: bool,
}

impl PrevCovspan {
    fn is_mergeable(&self, other: &CurrCovspan) -> bool {
        self.bcb == other.bcb && !self.is_hole && !other.is_hole
    }

    fn merge_from(&mut self, other: &CurrCovspan) {
        debug_assert!(self.is_mergeable(other));
        self.span = self.span.to(other.span);
        self.merged_spans.push(other.span);
    }

    fn cutoff_statements_at(mut self, cutoff_pos: BytePos) -> Option<RefinedCovspan> {
        self.merged_spans.retain(|span| span.hi() <= cutoff_pos);
        if let Some(max_hi) = self.merged_spans.iter().map(|span| span.hi()).max() {
            self.span = self.span.with_hi(max_hi);
        }

        if self.merged_spans.is_empty() { None } else { Some(self.into_refined()) }
    }

    fn refined_copy(&self) -> RefinedCovspan {
        let &Self { span, bcb, merged_spans: _, is_hole } = self;
        RefinedCovspan { span, bcb, is_hole }
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
    is_hole: bool,
}

impl RefinedCovspan {
    fn is_mergeable(&self, other: &Self) -> bool {
        self.bcb == other.bcb && !self.is_hole && !other.is_hole
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
///  * Carve out (leave uncovered) any "hole" spans that need to be left blank
///    (e.g. closures that will be counted by their own MIR body)
struct SpansRefiner {
    /// The initial set of coverage spans, sorted by `Span` (`lo` and `hi`) and by relative
    /// dominance between the `BasicCoverageBlock`s of equal `Span`s.
    sorted_spans_iter: std::vec::IntoIter<SpanFromMir>,

    /// The current coverage span to compare to its `prev`, to possibly merge, discard,
    /// or cause `prev` to be modified or discarded.
    /// If `curr` is not discarded or merged, it becomes `prev` for the next iteration.
    some_curr: Option<CurrCovspan>,

    /// The coverage span from a prior iteration; typically assigned from that iteration's `curr`.
    /// If that `curr` was discarded, `prev` retains its value from the previous iteration.
    some_prev: Option<PrevCovspan>,

    /// The final coverage spans to add to the coverage map. A `Counter` or `Expression`
    /// will also be injected into the MIR for each BCB that has associated spans.
    refined_spans: Vec<RefinedCovspan>,
}

impl SpansRefiner {
    /// Takes the initial list of (sorted) spans extracted from MIR, and "refines"
    /// them by merging compatible adjacent spans, removing redundant spans,
    /// and carving holes in spans when they overlap in unwanted ways.
    fn refine_sorted_spans(sorted_spans: Vec<SpanFromMir>) -> Vec<RefinedCovspan> {
        let sorted_spans_len = sorted_spans.len();
        let this = Self {
            sorted_spans_iter: sorted_spans.into_iter(),
            some_curr: None,
            some_prev: None,
            refined_spans: Vec::with_capacity(sorted_spans_len),
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
                debug!(?prev, "curr will be merged into prev");
                let curr = self.take_curr();
                self.prev_mut().merge_from(&curr);
            } else if prev.span.hi() <= curr.span.lo() {
                debug!(
                    "  different bcbs and disjoint spans, so keep curr for next iter, and add prev={prev:?}",
                );
                let prev = self.take_prev().into_refined();
                self.refined_spans.push(prev);
            } else if prev.is_hole {
                // drop any equal or overlapping span (`curr`) and keep `prev` to test again in the
                // next iter
                debug!(?prev, "prev (a hole) overlaps curr, so discarding curr");
                self.take_curr(); // Discards curr.
            } else if curr.is_hole {
                self.carve_out_span_for_hole();
            } else {
                self.cutoff_prev_at_overlapping_curr();
            }
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

        // Discard hole spans, since their purpose was to carve out chunks from
        // other spans, but we don't want the holes themselves in the final mappings.
        self.refined_spans.retain(|covspan| !covspan.is_hole);
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
                // a region of code, such as skipping past a hole.
                debug!(?prev, "prev.span starts after curr.span, so curr will be dropped");
            } else {
                self.some_curr = Some(CurrCovspan::new(curr.span, curr.bcb, curr.is_hole));
                return true;
            }
        }
        false
    }

    /// If `prev`s span extends left of the hole (`curr`), carve out the hole's span from
    /// `prev`'s span. Add the portion of the span to the left of the hole; and if the span
    /// extends to the right of the hole, update `prev` to that portion of the span.
    fn carve_out_span_for_hole(&mut self) {
        let prev = self.prev();
        let curr = self.curr();

        let left_cutoff = curr.span.lo();
        let right_cutoff = curr.span.hi();
        let has_pre_hole_span = prev.span.lo() < right_cutoff;
        let has_post_hole_span = prev.span.hi() > right_cutoff;

        if has_pre_hole_span {
            let mut pre_hole = prev.refined_copy();
            pre_hole.span = pre_hole.span.with_hi(left_cutoff);
            debug!(?pre_hole, "prev overlaps a hole; adding pre-hole span");
            self.refined_spans.push(pre_hole);
        }

        if has_post_hole_span {
            // Mutate `prev.span` to start after the hole (and discard curr).
            self.prev_mut().span = self.prev().span.with_lo(right_cutoff);
            debug!(prev=?self.prev(), "mutated prev to start after the hole");

            // Prevent this curr from becoming prev.
            let hole_covspan = self.take_curr().into_refined();
            self.refined_spans.push(hole_covspan); // since self.prev() was already updated
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

        let curr_span = self.curr().span;
        if let Some(prev) = self.take_prev().cutoff_statements_at(curr_span.lo()) {
            debug!("after cutoff, adding {prev:?}");
            self.refined_spans.push(prev);
        } else {
            debug!("prev was eliminated by cutoff");
        }
    }
}
