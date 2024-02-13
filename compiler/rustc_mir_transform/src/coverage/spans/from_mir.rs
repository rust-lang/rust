use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir::{
    self, AggregateKind, FakeReadCause, Rvalue, Statement, StatementKind, Terminator,
    TerminatorKind,
};
use rustc_span::{ExpnKind, MacroKind, Span, Symbol};

use crate::coverage::graph::{
    BasicCoverageBlock, BasicCoverageBlockData, CoverageGraph, START_BCB,
};
use crate::coverage::spans::CoverageSpan;
use crate::coverage::ExtractedHirInfo;

/// Traverses the MIR body to produce an initial collection of coverage-relevant
/// spans, each associated with a node in the coverage graph (BCB) and possibly
/// other metadata.
///
/// The returned spans are sorted in a specific order that is expected by the
/// subsequent span-refinement step.
pub(super) fn mir_to_initial_sorted_coverage_spans(
    mir_body: &mir::Body<'_>,
    hir_info: &ExtractedHirInfo,
    basic_coverage_blocks: &CoverageGraph,
) -> Vec<CoverageSpan> {
    let &ExtractedHirInfo { body_span, .. } = hir_info;

    let mut initial_spans = vec![];

    for (bcb, bcb_data) in basic_coverage_blocks.iter_enumerated() {
        initial_spans.extend(bcb_to_initial_coverage_spans(mir_body, body_span, bcb, bcb_data));
    }

    // Only add the signature span if we found at least one span in the body.
    if !initial_spans.is_empty() {
        // If there is no usable signature span, add a fake one (before refinement)
        // to avoid an ugly gap between the body start and the first real span.
        // FIXME: Find a more principled way to solve this problem.
        let fn_sig_span = hir_info.fn_sig_span_extended.unwrap_or_else(|| body_span.shrink_to_lo());
        initial_spans.push(SpanFromMir::for_fn_sig(fn_sig_span));
    }

    initial_spans.sort_by(|a, b| basic_coverage_blocks.cmp_in_dominator_order(a.bcb, b.bcb));
    remove_unwanted_macro_spans(&mut initial_spans);
    split_visible_macro_spans(&mut initial_spans);

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

    initial_spans.into_iter().map(SpanFromMir::into_coverage_span).collect::<Vec<_>>()
}

/// Macros that expand into branches (e.g. `assert!`, `trace!`) tend to generate
/// multiple condition/consequent blocks that have the span of the whole macro
/// invocation, which is unhelpful. Keeping only the first such span seems to
/// give better mappings, so remove the others.
///
/// (The input spans should be sorted in BCB dominator order, so that the
/// retained "first" span is likely to dominate the others.)
fn remove_unwanted_macro_spans(initial_spans: &mut Vec<SpanFromMir>) {
    let mut seen_macro_spans = FxHashSet::default();
    initial_spans.retain(|covspan| {
        // Ignore (retain) closure spans and non-macro-expansion spans.
        if covspan.is_closure || covspan.visible_macro.is_none() {
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
fn split_visible_macro_spans(initial_spans: &mut Vec<SpanFromMir>) {
    let mut extra_spans = vec![];

    initial_spans.retain(|covspan| {
        if covspan.is_closure {
            return true;
        }

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

        assert!(!covspan.is_closure);
        extra_spans.push(SpanFromMir::new(before, covspan.visible_macro, covspan.bcb, false));
        extra_spans.push(SpanFromMir::new(after, covspan.visible_macro, covspan.bcb, false));
        false // Discard the original covspan that we just split.
    });

    // The newly-split spans are added at the end, so any previous sorting
    // is not preserved.
    initial_spans.extend(extra_spans);
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
) -> impl Iterator<Item = SpanFromMir> + Captures<'a> + Captures<'tcx> {
    bcb_data.basic_blocks.iter().flat_map(move |&bb| {
        let data = &mir_body[bb];

        let statement_spans = data.statements.iter().filter_map(move |statement| {
            let expn_span = filtered_statement_span(statement)?;
            let (span, visible_macro) =
                unexpand_into_body_span_with_visible_macro(expn_span, body_span)?;

            Some(SpanFromMir::new(span, visible_macro, bcb, is_closure_like(statement)))
        });

        let terminator_span = Some(data.terminator()).into_iter().filter_map(move |terminator| {
            let expn_span = filtered_terminator_span(terminator)?;
            let (span, visible_macro) =
                unexpand_into_body_span_with_visible_macro(expn_span, body_span)?;

            Some(SpanFromMir::new(span, visible_macro, bcb, false))
        });

        statement_spans.chain(terminator_span)
    })
}

fn is_closure_like(statement: &Statement<'_>) -> bool {
    match statement.kind {
        StatementKind::Assign(box (_, Rvalue::Aggregate(box ref agg_kind, _))) => match agg_kind {
            AggregateKind::Closure(_, _)
            | AggregateKind::Coroutine(_, _)
            | AggregateKind::CoroutineClosure(..) => true,
            _ => false,
        },
        _ => false,
    }
}

/// If the MIR `Statement` has a span contributive to computing coverage spans,
/// return it; otherwise return `None`.
fn filtered_statement_span(statement: &Statement<'_>) -> Option<Span> {
    use mir::coverage::CoverageKind;

    match statement.kind {
        // These statements have spans that are often outside the scope of the executed source code
        // for their parent `BasicBlock`.
        StatementKind::StorageLive(_)
        | StatementKind::StorageDead(_)
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

        // Retain spans from most other statements.
        StatementKind::FakeRead(box (_, _)) // Not including `ForGuardBinding`
        | StatementKind::Intrinsic(..)
        | StatementKind::Coverage(box mir::Coverage {
            // The purpose of `SpanMarker` is to be matched and accepted here.
            kind: CoverageKind::SpanMarker
        })
        | StatementKind::Assign(_)
        | StatementKind::SetDiscriminant { .. }
        | StatementKind::Deinit(..)
        | StatementKind::Retag(_, _)
        | StatementKind::PlaceMention(..)
        | StatementKind::AscribeUserType(_, _) => {
            Some(statement.source_info.span)
        }

        StatementKind::Coverage(box mir::Coverage {
            // These coverage statements should not exist prior to coverage instrumentation.
            kind: CoverageKind::CounterIncrement { .. } | CoverageKind::ExpressionUsed { .. }
        }) => bug!("Unexpected coverage statement found during coverage instrumentation: {statement:?}"),
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
/// contained within the `body_span`, `None` is returned.
///
/// [^1]Expansions result from Rust syntax including macros, syntactic sugar,
/// etc.).
fn unexpand_into_body_span_with_visible_macro(
    original_span: Span,
    body_span: Span,
) -> Option<(Span, Option<Symbol>)> {
    let (span, prev) = unexpand_into_body_span_with_prev(original_span, body_span)?;

    let visible_macro = prev
        .map(|prev| match prev.ctxt().outer_expn_data().kind {
            ExpnKind::Macro(MacroKind::Bang, name) => Some(name),
            _ => None,
        })
        .flatten();

    Some((span, visible_macro))
}

/// Walks through the expansion ancestors of `original_span` to find a span that
/// is contained in `body_span` and has the same [`SyntaxContext`] as `body_span`.
/// The ancestor that was traversed just before the matching span (if any) is
/// also returned.
///
/// For example, a return value of `Some((ancestor, Some(prev))` means that:
/// - `ancestor == original_span.find_ancestor_inside_same_ctxt(body_span)`
/// - `ancestor == prev.parent_callsite()`
///
/// [`SyntaxContext`]: rustc_span::SyntaxContext
fn unexpand_into_body_span_with_prev(
    original_span: Span,
    body_span: Span,
) -> Option<(Span, Option<Span>)> {
    let mut prev = None;
    let mut curr = original_span;

    while !body_span.contains(curr) || !curr.eq_ctxt(body_span) {
        prev = Some(curr);
        curr = curr.parent_callsite()?;
    }

    debug_assert_eq!(Some(curr), original_span.find_ancestor_in_same_ctxt(body_span));
    if let Some(prev) = prev {
        debug_assert_eq!(Some(curr), prev.parent_callsite());
    }

    Some((curr, prev))
}

#[derive(Debug)]
struct SpanFromMir {
    /// A span that has been extracted from MIR and then "un-expanded" back to
    /// within the current function's `body_span`. After various intermediate
    /// processing steps, this span is emitted as part of the final coverage
    /// mappings.
    ///
    /// With the exception of `fn_sig_span`, this should always be contained
    /// within `body_span`.
    span: Span,
    visible_macro: Option<Symbol>,
    bcb: BasicCoverageBlock,
    is_closure: bool,
}

impl SpanFromMir {
    fn for_fn_sig(fn_sig_span: Span) -> Self {
        Self::new(fn_sig_span, None, START_BCB, false)
    }

    fn new(
        span: Span,
        visible_macro: Option<Symbol>,
        bcb: BasicCoverageBlock,
        is_closure: bool,
    ) -> Self {
        Self { span, visible_macro, bcb, is_closure }
    }

    fn into_coverage_span(self) -> CoverageSpan {
        let Self { span, visible_macro: _, bcb, is_closure } = self;
        CoverageSpan::new(span, bcb, is_closure)
    }
}
