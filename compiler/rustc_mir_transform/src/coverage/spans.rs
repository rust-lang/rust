use rustc_middle::mir;
use rustc_span::Span;

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
    let buckets =
        from_mir::mir_to_initial_sorted_coverage_spans(mir_body, hir_info, basic_coverage_blocks);
    for covspans in buckets {
        let covspans = refine_sorted_spans(covspans);
        code_mappings.extend(covspans.into_iter().map(|RefinedCovspan { span, bcb }| {
            // Each span produced by the refiner represents an ordinary code region.
            mappings::CodeMapping { span, bcb }
        }));
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
