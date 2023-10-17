use rustc_data_structures::captures::Captures;
use rustc_middle::mir::{
    self, AggregateKind, FakeReadCause, Rvalue, Statement, StatementKind, Terminator,
    TerminatorKind,
};
use rustc_span::Span;

use crate::coverage::graph::{BasicCoverageBlock, BasicCoverageBlockData, CoverageGraph};
use crate::coverage::spans::CoverageSpan;

pub(super) fn mir_to_initial_sorted_coverage_spans(
    mir_body: &mir::Body<'_>,
    fn_sig_span: Span,
    body_span: Span,
    basic_coverage_blocks: &CoverageGraph,
) -> Vec<CoverageSpan> {
    let mut initial_spans = Vec::with_capacity(mir_body.basic_blocks.len() * 2);
    for (bcb, bcb_data) in basic_coverage_blocks.iter_enumerated() {
        initial_spans.extend(bcb_to_initial_coverage_spans(mir_body, body_span, bcb, bcb_data));
    }

    if initial_spans.is_empty() {
        // This can happen if, for example, the function is unreachable (contains only a
        // `BasicBlock`(s) with an `Unreachable` terminator).
        return initial_spans;
    }

    initial_spans.push(CoverageSpan::for_fn_sig(fn_sig_span));

    initial_spans.sort_by(|a, b| {
        // First sort by span start.
        Ord::cmp(&a.span.lo(), &b.span.lo())
            // If span starts are the same, sort by span end in reverse order.
            // This ensures that if spans A and B are adjacent in the list,
            // and they overlap but are not equal, then either:
            // - Span A extends further left, or
            // - Both have the same start and span A extends further right
            .then_with(|| Ord::cmp(&a.span.hi(), &b.span.hi()).reverse())
            // If both spans are equal, sort the BCBs in dominator order,
            // so that dominating BCBs come before other BCBs they dominate.
            .then_with(|| basic_coverage_blocks.cmp_in_dominator_order(a.bcb, b.bcb))
            // If two spans are otherwise identical, put closure spans first,
            // as this seems to be what the refinement step expects.
            .then_with(|| Ord::cmp(&a.is_closure, &b.is_closure).reverse())
    });

    initial_spans
}

// Generate a set of `CoverageSpan`s from the filtered set of `Statement`s and `Terminator`s of
// the `BasicBlock`(s) in the given `BasicCoverageBlockData`. One `CoverageSpan` is generated
// for each `Statement` and `Terminator`. (Note that subsequent stages of coverage analysis will
// merge some `CoverageSpan`s, at which point a `CoverageSpan` may represent multiple
// `Statement`s and/or `Terminator`s.)
fn bcb_to_initial_coverage_spans<'a, 'tcx>(
    mir_body: &'a mir::Body<'tcx>,
    body_span: Span,
    bcb: BasicCoverageBlock,
    bcb_data: &'a BasicCoverageBlockData,
) -> impl Iterator<Item = CoverageSpan> + Captures<'a> + Captures<'tcx> {
    bcb_data.basic_blocks.iter().flat_map(move |&bb| {
        let data = &mir_body[bb];

        let statement_spans = data.statements.iter().filter_map(move |statement| {
            let expn_span = filtered_statement_span(statement)?;
            let span = function_source_span(expn_span, body_span);

            Some(CoverageSpan::new(span, expn_span, bcb, is_closure(statement)))
        });

        let terminator_span = Some(data.terminator()).into_iter().filter_map(move |terminator| {
            let expn_span = filtered_terminator_span(terminator)?;
            let span = function_source_span(expn_span, body_span);

            Some(CoverageSpan::new(span, expn_span, bcb, false))
        });

        statement_spans.chain(terminator_span)
    })
}

fn is_closure(statement: &Statement<'_>) -> bool {
    match statement.kind {
        StatementKind::Assign(box (_, Rvalue::Aggregate(box ref agg_kind, _))) => match agg_kind {
            AggregateKind::Closure(_, _) | AggregateKind::Coroutine(_, _, _) => true,
            _ => false,
        },
        _ => false,
    }
}

/// If the MIR `Statement` has a span contributive to computing coverage spans,
/// return it; otherwise return `None`.
fn filtered_statement_span(statement: &Statement<'_>) -> Option<Span> {
    match statement.kind {
        // These statements have spans that are often outside the scope of the executed source code
        // for their parent `BasicBlock`.
        StatementKind::StorageLive(_)
        | StatementKind::StorageDead(_)
        // Coverage should not be encountered, but don't inject coverage coverage
        | StatementKind::Coverage(_)
        // Ignore `ConstEvalCounter`s
        | StatementKind::ConstEvalCounter
        // Ignore `Nop`s
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

        // Retain spans from all other statements
        StatementKind::FakeRead(box (_, _)) // Not including `ForGuardBinding`
        | StatementKind::Intrinsic(..)
        | StatementKind::Assign(_)
        | StatementKind::SetDiscriminant { .. }
        | StatementKind::Deinit(..)
        | StatementKind::Retag(_, _)
        | StatementKind::PlaceMention(..)
        | StatementKind::AscribeUserType(_, _) => {
            Some(statement.source_info.span)
        }
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
        TerminatorKind::Unreachable // Unreachable blocks are not connected to the MIR CFG
        | TerminatorKind::Assert { .. }
        | TerminatorKind::Drop { .. }
        | TerminatorKind::SwitchInt { .. }
        // For `FalseEdge`, only the `real` branch is taken, so it is similar to a `Goto`.
        | TerminatorKind::FalseEdge { .. }
        | TerminatorKind::Goto { .. } => None,

        // Call `func` operand can have a more specific span when part of a chain of calls
        | TerminatorKind::Call { ref func, .. } => {
            let mut span = terminator.source_info.span;
            if let mir::Operand::Constant(box constant) = func {
                if constant.span.lo() > span.lo() {
                    span = span.with_lo(constant.span.lo());
                }
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
        | TerminatorKind::InlineAsm { .. } => {
            Some(terminator.source_info.span)
        }
    }
}

/// Returns an extrapolated span (pre-expansion[^1]) corresponding to a range
/// within the function's body source. This span is guaranteed to be contained
/// within, or equal to, the `body_span`. If the extrapolated span is not
/// contained within the `body_span`, the `body_span` is returned.
///
/// [^1]Expansions result from Rust syntax including macros, syntactic sugar,
/// etc.).
#[inline]
fn function_source_span(span: Span, body_span: Span) -> Span {
    use rustc_span::source_map::original_sp;

    let original_span = original_sp(span, body_span).with_ctxt(body_span.ctxt());
    if body_span.contains(original_span) { original_span } else { body_span }
}
