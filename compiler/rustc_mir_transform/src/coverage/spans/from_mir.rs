use std::iter;

use rustc_middle::bug;
use rustc_middle::mir::coverage::CoverageKind;
use rustc_middle::mir::{
    self, FakeReadCause, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_span::Span;

use crate::coverage::graph::{BasicCoverageBlock, CoverageGraph};

#[derive(Debug)]
pub(crate) struct RawSpanFromMir {
    /// A span that has been extracted from a MIR statement/terminator, but
    /// hasn't been "unexpanded", so it might not lie within the function body
    /// span and might be part of an expansion with a different context.
    pub(crate) raw_span: Span,
    pub(crate) bcb: BasicCoverageBlock,
}

/// Generates an initial set of coverage spans from the statements and
/// terminators in the function's MIR body, each associated with its
/// corresponding node in the coverage graph.
///
/// This is necessarily an inexact process, because MIR isn't designed to
/// capture source spans at the level of detail we would want for coverage,
/// but it's good enough to be better than nothing.
pub(crate) fn extract_raw_spans_from_mir<'tcx>(
    mir_body: &mir::Body<'tcx>,
    graph: &CoverageGraph,
) -> Vec<RawSpanFromMir> {
    let mut raw_spans = vec![];

    // We only care about blocks that are part of the coverage graph.
    for (bcb, bcb_data) in graph.iter_enumerated() {
        let make_raw_span = |raw_span: Span| RawSpanFromMir { raw_span, bcb };

        // A coverage graph node can consist of multiple basic blocks.
        for &bb in &bcb_data.basic_blocks {
            let bb_data = &mir_body[bb];

            let statements = bb_data.statements.iter();
            raw_spans.extend(statements.filter_map(filtered_statement_span).map(make_raw_span));

            // There's only one terminator, but wrap it in an iterator to
            // mirror the handling of statements.
            let terminator = iter::once(bb_data.terminator());
            raw_spans.extend(terminator.filter_map(filtered_terminator_span).map(make_raw_span));
        }
    }

    raw_spans
}

/// If the MIR `Statement` has a span contributive to computing coverage spans,
/// return it; otherwise return `None`.
fn filtered_statement_span(statement: &Statement<'_>) -> Option<Span> {
    match statement.kind {
        // These statements have spans that are often outside the scope of the executed source code
        // for their parent `BasicBlock`.
        StatementKind::StorageLive(_)
        | StatementKind::StorageDead(_)
        | StatementKind::ConstEvalCounter
        | StatementKind::BackwardIncompatibleDropHint { .. }
        | StatementKind::Nop => None,

        // FIXME(#78546): MIR InstrumentCoverage - Can the source_info.span for `FakeRead`
        // statements be more consistent?
        //
        // FakeReadCause::ForGuardBinding, in this example:
        //     match somenum {
        //         x if x < 1 => { ... }
        //     }...
        // The BasicBlock within the match arm code included one of these statements, but the span
        // for it covered the `1` in this source. The actual statements have nothing to do with that
        // source span:
        //     FakeRead(ForGuardBinding, _4);
        // where `_4` is:
        //     _4 = &_1; (at the span for the first `x`)
        // and `_1` is the `Place` for `somenum`.
        //
        // If and when the Issue is resolved, remove this special case match pattern:
        StatementKind::FakeRead(box (FakeReadCause::ForGuardBinding, _)) => None,

        // Retain spans from most other statements.
        StatementKind::FakeRead(_)
        | StatementKind::Intrinsic(..)
        | StatementKind::Coverage(
            // The purpose of `SpanMarker` is to be matched and accepted here.
            CoverageKind::SpanMarker,
        )
        | StatementKind::Assign(_)
        | StatementKind::SetDiscriminant { .. }
        | StatementKind::Deinit(..)
        | StatementKind::Retag(_, _)
        | StatementKind::PlaceMention(..)
        | StatementKind::AscribeUserType(_, _) => Some(statement.source_info.span),

        // Block markers are used for branch coverage, so ignore them here.
        StatementKind::Coverage(CoverageKind::BlockMarker { .. }) => None,

        // These coverage statements should not exist prior to coverage instrumentation.
        StatementKind::Coverage(CoverageKind::VirtualCounter { .. }) => bug!(
            "Unexpected coverage statement found during coverage instrumentation: {statement:?}"
        ),
    }
}

/// If the MIR `Terminator` has a span contributive to computing coverage spans,
/// return it; otherwise return `None`.
fn filtered_terminator_span(terminator: &Terminator<'_>) -> Option<Span> {
    match terminator.kind {
        // These terminators have spans that don't positively contribute to computing a reasonable
        // span of actually executed source code. (For example, SwitchInt terminators extracted from
        // an `if condition { block }` has a span that includes the executed block, if true,
        // but for coverage, the code region executed, up to *and* through the SwitchInt,
        // actually stops before the if's block.)
        TerminatorKind::Unreachable
        | TerminatorKind::Assert { .. }
        | TerminatorKind::Drop { .. }
        | TerminatorKind::SwitchInt { .. }
        | TerminatorKind::FalseEdge { .. }
        | TerminatorKind::Goto { .. } => None,

        // Call `func` operand can have a more specific span when part of a chain of calls
        TerminatorKind::Call { ref func, .. } | TerminatorKind::TailCall { ref func, .. } => {
            let mut span = terminator.source_info.span;
            if let mir::Operand::Constant(constant) = func
                && span.contains(constant.span)
            {
                span = constant.span;
            }
            Some(span)
        }

        // Retain spans from all other terminators
        TerminatorKind::UnwindResume
        | TerminatorKind::UnwindTerminate(_)
        | TerminatorKind::Return
        | TerminatorKind::Yield { .. }
        | TerminatorKind::CoroutineDrop
        | TerminatorKind::FalseUnwind { .. }
        | TerminatorKind::InlineAsm { .. } => Some(terminator.source_info.span),
    }
}

#[derive(Debug)]
pub(crate) struct Hole {
    pub(crate) span: Span,
}

impl Hole {
    pub(crate) fn merge_if_overlapping_or_adjacent(&mut self, other: &mut Self) -> bool {
        if !self.span.overlaps_or_adjacent(other.span) {
            return false;
        }

        self.span = self.span.to(other.span);
        true
    }
}
