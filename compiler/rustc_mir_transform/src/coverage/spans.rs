use rustc_middle::bug;
use rustc_middle::mir;
use rustc_span::{BytePos, Span};

use crate::coverage::graph::{BasicCoverageBlock, CoverageGraph};
use crate::coverage::mappings;
use crate::coverage::spans::from_mir::SpanFromMir;
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
    let sorted_span_buckets =
        from_mir::mir_to_initial_sorted_coverage_spans(mir_body, hir_info, basic_coverage_blocks);
    for bucket in sorted_span_buckets {
        let refined_spans = SpansRefiner::refine_sorted_spans(bucket);
        code_mappings.extend(refined_spans.into_iter().map(|RefinedCovspan { span, bcb }| {
            // Each span produced by the refiner represents an ordinary code region.
            mappings::CodeMapping { span, bcb }
        }));
    }
}

#[derive(Debug)]
struct CurrCovspan {
    span: Span,
    bcb: BasicCoverageBlock,
}

impl CurrCovspan {
    fn new(span: Span, bcb: BasicCoverageBlock) -> Self {
        Self { span, bcb }
    }

    fn into_prev(self) -> PrevCovspan {
        let Self { span, bcb } = self;
        PrevCovspan { span, bcb, merged_spans: vec![span] }
    }
}

#[derive(Debug)]
struct PrevCovspan {
    span: Span,
    bcb: BasicCoverageBlock,
    /// List of all the original spans from MIR that have been merged into this
    /// span. Mainly used to precisely skip over gaps when truncating a span.
    merged_spans: Vec<Span>,
}

impl PrevCovspan {
    fn is_mergeable(&self, other: &CurrCovspan) -> bool {
        self.bcb == other.bcb
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

    fn into_refined(self) -> RefinedCovspan {
        let Self { span, bcb, merged_spans: _ } = self;
        RefinedCovspan { span, bcb }
    }
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

/// Converts the initial set of coverage spans (one per MIR `Statement` or `Terminator`) into a
/// minimal set of coverage spans, using the BCB CFG to determine where it is safe and useful to:
///
///  * Remove duplicate source code coverage regions
///  * Merge spans that represent continuous (both in source code and control flow), non-branching
///    execution
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
        if let Some(SpanFromMir { span, bcb, .. }) = self.sorted_spans_iter.next() {
            // This code only sees sorted spans after hole-carving, so there should
            // be no way for `curr` to start before `prev`.
            if let Some(prev) = &self.some_prev {
                debug_assert!(prev.span.lo() <= span.lo());
            }
            self.some_curr = Some(CurrCovspan::new(span, bcb));
            debug!(?self.some_prev, ?self.some_curr, "next_coverage_span");
            true
        } else {
            false
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
