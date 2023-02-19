use super::debug::term_type;
use super::graph::{BasicCoverageBlock, BasicCoverageBlockData, CoverageGraph, START_BCB};

use itertools::Itertools;
use rustc_data_structures::graph::WithNumNodes;
use rustc_middle::mir::spanview::source_range_no_file;
use rustc_middle::mir::{
    self, AggregateKind, BasicBlock, FakeReadCause, Rvalue, Statement, StatementKind, Terminator,
    TerminatorKind,
};
use rustc_middle::ty::TyCtxt;
use rustc_span::source_map::original_sp;
use rustc_span::{BytePos, ExpnKind, MacroKind, Span, Symbol};

use std::cell::RefCell;
use std::cmp::Ordering;

#[derive(Debug, Copy, Clone)]
pub(super) enum CoverageStatement {
    Statement(BasicBlock, Span, usize),
    Terminator(BasicBlock, Span),
}

impl CoverageStatement {
    pub fn format<'tcx>(&self, tcx: TyCtxt<'tcx>, mir_body: &mir::Body<'tcx>) -> String {
        match *self {
            Self::Statement(bb, span, stmt_index) => {
                let stmt = &mir_body[bb].statements[stmt_index];
                format!(
                    "{}: @{}[{}]: {:?}",
                    source_range_no_file(tcx, span),
                    bb.index(),
                    stmt_index,
                    stmt
                )
            }
            Self::Terminator(bb, span) => {
                let term = mir_body[bb].terminator();
                format!(
                    "{}: @{}.{}: {:?}",
                    source_range_no_file(tcx, span),
                    bb.index(),
                    term_type(&term.kind),
                    term.kind
                )
            }
        }
    }

    pub fn span(&self) -> Span {
        match self {
            Self::Statement(_, span, _) | Self::Terminator(_, span) => *span,
        }
    }
}

/// A BCB is deconstructed into one or more `Span`s. Each `Span` maps to a `CoverageSpan` that
/// references the originating BCB and one or more MIR `Statement`s and/or `Terminator`s.
/// Initially, the `Span`s come from the `Statement`s and `Terminator`s, but subsequent
/// transforms can combine adjacent `Span`s and `CoverageSpan` from the same BCB, merging the
/// `CoverageStatement` vectors, and the `Span`s to cover the extent of the combined `Span`s.
///
/// Note: A `CoverageStatement` merged into another CoverageSpan may come from a `BasicBlock` that
/// is not part of the `CoverageSpan` bcb if the statement was included because it's `Span` matches
/// or is subsumed by the `Span` associated with this `CoverageSpan`, and it's `BasicBlock`
/// `dominates()` the `BasicBlock`s in this `CoverageSpan`.
#[derive(Debug, Clone)]
pub(super) struct CoverageSpan {
    pub span: Span,
    pub expn_span: Span,
    pub current_macro_or_none: RefCell<Option<Option<Symbol>>>,
    pub bcb: BasicCoverageBlock,
    pub coverage_statements: Vec<CoverageStatement>,
    pub is_closure: bool,
}

impl CoverageSpan {
    pub fn for_fn_sig(fn_sig_span: Span) -> Self {
        Self {
            span: fn_sig_span,
            expn_span: fn_sig_span,
            current_macro_or_none: Default::default(),
            bcb: START_BCB,
            coverage_statements: vec![],
            is_closure: false,
        }
    }

    pub fn for_statement(
        statement: &Statement<'_>,
        span: Span,
        expn_span: Span,
        bcb: BasicCoverageBlock,
        bb: BasicBlock,
        stmt_index: usize,
    ) -> Self {
        let is_closure = match statement.kind {
            StatementKind::Assign(box (_, Rvalue::Aggregate(box ref kind, _))) => {
                matches!(kind, AggregateKind::Closure(_, _) | AggregateKind::Generator(_, _, _))
            }
            _ => false,
        };

        Self {
            span,
            expn_span,
            current_macro_or_none: Default::default(),
            bcb,
            coverage_statements: vec![CoverageStatement::Statement(bb, span, stmt_index)],
            is_closure,
        }
    }

    pub fn for_terminator(
        span: Span,
        expn_span: Span,
        bcb: BasicCoverageBlock,
        bb: BasicBlock,
    ) -> Self {
        Self {
            span,
            expn_span,
            current_macro_or_none: Default::default(),
            bcb,
            coverage_statements: vec![CoverageStatement::Terminator(bb, span)],
            is_closure: false,
        }
    }

    pub fn merge_from(&mut self, mut other: CoverageSpan) {
        debug_assert!(self.is_mergeable(&other));
        self.span = self.span.to(other.span);
        self.coverage_statements.append(&mut other.coverage_statements);
    }

    pub fn cutoff_statements_at(&mut self, cutoff_pos: BytePos) {
        self.coverage_statements.retain(|covstmt| covstmt.span().hi() <= cutoff_pos);
        if let Some(highest_covstmt) =
            self.coverage_statements.iter().max_by_key(|covstmt| covstmt.span().hi())
        {
            self.span = self.span.with_hi(highest_covstmt.span().hi());
        }
    }

    #[inline]
    pub fn is_mergeable(&self, other: &Self) -> bool {
        self.is_in_same_bcb(other) && !(self.is_closure || other.is_closure)
    }

    #[inline]
    pub fn is_in_same_bcb(&self, other: &Self) -> bool {
        self.bcb == other.bcb
    }

    pub fn format<'tcx>(&self, tcx: TyCtxt<'tcx>, mir_body: &mir::Body<'tcx>) -> String {
        format!(
            "{}\n    {}",
            source_range_no_file(tcx, self.span),
            self.format_coverage_statements(tcx, mir_body).replace('\n', "\n    "),
        )
    }

    pub fn format_coverage_statements<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        mir_body: &mir::Body<'tcx>,
    ) -> String {
        let mut sorted_coverage_statements = self.coverage_statements.clone();
        sorted_coverage_statements.sort_unstable_by_key(|covstmt| match *covstmt {
            CoverageStatement::Statement(bb, _, index) => (bb, index),
            CoverageStatement::Terminator(bb, _) => (bb, usize::MAX),
        });
        sorted_coverage_statements.iter().map(|covstmt| covstmt.format(tcx, mir_body)).join("\n")
    }

    /// If the span is part of a macro, returns the macro name symbol.
    pub fn current_macro(&self) -> Option<Symbol> {
        self.current_macro_or_none
            .borrow_mut()
            .get_or_insert_with(|| {
                if let ExpnKind::Macro(MacroKind::Bang, current_macro) =
                    self.expn_span.ctxt().outer_expn_data().kind
                {
                    return Some(current_macro);
                }
                None
            })
            .map(|symbol| symbol)
    }

    /// If the span is part of a macro, and the macro is visible (expands directly to the given
    /// body_span), returns the macro name symbol.
    pub fn visible_macro(&self, body_span: Span) -> Option<Symbol> {
        if let Some(current_macro) = self.current_macro() && self
            .expn_span
            .parent_callsite()
            .unwrap_or_else(|| bug!("macro must have a parent"))
            .eq_ctxt(body_span)
        {
            return Some(current_macro);
        }
        None
    }

    pub fn is_macro_expansion(&self) -> bool {
        self.current_macro().is_some()
    }
}

/// Converts the initial set of `CoverageSpan`s (one per MIR `Statement` or `Terminator`) into a
/// minimal set of `CoverageSpan`s, using the BCB CFG to determine where it is safe and useful to:
///
///  * Remove duplicate source code coverage regions
///  * Merge spans that represent continuous (both in source code and control flow), non-branching
///    execution
///  * Carve out (leave uncovered) any span that will be counted by another MIR (notably, closures)
pub struct CoverageSpans<'a, 'tcx> {
    /// The MIR, used to look up `BasicBlockData`.
    mir_body: &'a mir::Body<'tcx>,

    /// A `Span` covering the signature of function for the MIR.
    fn_sig_span: Span,

    /// A `Span` covering the function body of the MIR (typically from left curly brace to right
    /// curly brace).
    body_span: Span,

    /// The BasicCoverageBlock Control Flow Graph (BCB CFG).
    basic_coverage_blocks: &'a CoverageGraph,

    /// The initial set of `CoverageSpan`s, sorted by `Span` (`lo` and `hi`) and by relative
    /// dominance between the `BasicCoverageBlock`s of equal `Span`s.
    sorted_spans_iter: Option<std::vec::IntoIter<CoverageSpan>>,

    /// The current `CoverageSpan` to compare to its `prev`, to possibly merge, discard, force the
    /// discard of the `prev` (and or `pending_dups`), or keep both (with `prev` moved to
    /// `pending_dups`). If `curr` is not discarded or merged, it becomes `prev` for the next
    /// iteration.
    some_curr: Option<CoverageSpan>,

    /// The original `span` for `curr`, in case `curr.span()` is modified. The `curr_original_span`
    /// **must not be mutated** (except when advancing to the next `curr`), even if `curr.span()`
    /// is mutated.
    curr_original_span: Span,

    /// The CoverageSpan from a prior iteration; typically assigned from that iteration's `curr`.
    /// If that `curr` was discarded, `prev` retains its value from the previous iteration.
    some_prev: Option<CoverageSpan>,

    /// Assigned from `curr_original_span` from the previous iteration. The `prev_original_span`
    /// **must not be mutated** (except when advancing to the next `prev`), even if `prev.span()`
    /// is mutated.
    prev_original_span: Span,

    /// A copy of the expn_span from the prior iteration.
    prev_expn_span: Option<Span>,

    /// One or more `CoverageSpan`s with the same `Span` but different `BasicCoverageBlock`s, and
    /// no `BasicCoverageBlock` in this list dominates another `BasicCoverageBlock` in the list.
    /// If a new `curr` span also fits this criteria (compared to an existing list of
    /// `pending_dups`), that `curr` `CoverageSpan` moves to `prev` before possibly being added to
    /// the `pending_dups` list, on the next iteration. As a result, if `prev` and `pending_dups`
    /// have the same `Span`, the criteria for `pending_dups` holds for `prev` as well: a `prev`
    /// with a matching `Span` does not dominate any `pending_dup` and no `pending_dup` dominates a
    /// `prev` with a matching `Span`)
    pending_dups: Vec<CoverageSpan>,

    /// The final `CoverageSpan`s to add to the coverage map. A `Counter` or `Expression`
    /// will also be injected into the MIR for each `CoverageSpan`.
    refined_spans: Vec<CoverageSpan>,
}

impl<'a, 'tcx> CoverageSpans<'a, 'tcx> {
    /// Generate a minimal set of `CoverageSpan`s, each representing a contiguous code region to be
    /// counted.
    ///
    /// The basic steps are:
    ///
    /// 1. Extract an initial set of spans from the `Statement`s and `Terminator`s of each
    ///    `BasicCoverageBlockData`.
    /// 2. Sort the spans by span.lo() (starting position). Spans that start at the same position
    ///    are sorted with longer spans before shorter spans; and equal spans are sorted
    ///    (deterministically) based on "dominator" relationship (if any).
    /// 3. Traverse the spans in sorted order to identify spans that can be dropped (for instance,
    ///    if another span or spans are already counting the same code region), or should be merged
    ///    into a broader combined span (because it represents a contiguous, non-branching, and
    ///    uninterrupted region of source code).
    ///
    ///    Closures are exposed in their enclosing functions as `Assign` `Rvalue`s, and since
    ///    closures have their own MIR, their `Span` in their enclosing function should be left
    ///    "uncovered".
    ///
    /// Note the resulting vector of `CoverageSpan`s may not be fully sorted (and does not need
    /// to be).
    pub(super) fn generate_coverage_spans(
        mir_body: &'a mir::Body<'tcx>,
        fn_sig_span: Span, // Ensured to be same SourceFile and SyntaxContext as `body_span`
        body_span: Span,
        basic_coverage_blocks: &'a CoverageGraph,
    ) -> Vec<CoverageSpan> {
        let mut coverage_spans = CoverageSpans {
            mir_body,
            fn_sig_span,
            body_span,
            basic_coverage_blocks,
            sorted_spans_iter: None,
            refined_spans: Vec::with_capacity(basic_coverage_blocks.num_nodes() * 2),
            some_curr: None,
            curr_original_span: Span::with_root_ctxt(BytePos(0), BytePos(0)),
            some_prev: None,
            prev_original_span: Span::with_root_ctxt(BytePos(0), BytePos(0)),
            prev_expn_span: None,
            pending_dups: Vec::new(),
        };

        let sorted_spans = coverage_spans.mir_to_initial_sorted_coverage_spans();

        coverage_spans.sorted_spans_iter = Some(sorted_spans.into_iter());

        coverage_spans.to_refined_spans()
    }

    fn mir_to_initial_sorted_coverage_spans(&self) -> Vec<CoverageSpan> {
        let mut initial_spans =
            Vec::<CoverageSpan>::with_capacity(self.mir_body.basic_blocks.len() * 2);
        for (bcb, bcb_data) in self.basic_coverage_blocks.iter_enumerated() {
            initial_spans.extend(self.bcb_to_initial_coverage_spans(bcb, bcb_data));
        }

        if initial_spans.is_empty() {
            // This can happen if, for example, the function is unreachable (contains only a
            // `BasicBlock`(s) with an `Unreachable` terminator).
            return initial_spans;
        }

        initial_spans.push(CoverageSpan::for_fn_sig(self.fn_sig_span));

        initial_spans.sort_unstable_by(|a, b| {
            if a.span.lo() == b.span.lo() {
                if a.span.hi() == b.span.hi() {
                    if a.is_in_same_bcb(b) {
                        Some(Ordering::Equal)
                    } else {
                        // Sort equal spans by dominator relationship (so dominators always come
                        // before the dominated equal spans). When later comparing two spans in
                        // order, the first will either dominate the second, or they will have no
                        // dominator relationship.
                        self.basic_coverage_blocks.dominators().rank_partial_cmp(a.bcb, b.bcb)
                    }
                } else {
                    // Sort hi() in reverse order so shorter spans are attempted after longer spans.
                    // This guarantees that, if a `prev` span overlaps, and is not equal to, a
                    // `curr` span, the prev span either extends further left of the curr span, or
                    // they start at the same position and the prev span extends further right of
                    // the end of the curr span.
                    b.span.hi().partial_cmp(&a.span.hi())
                }
            } else {
                a.span.lo().partial_cmp(&b.span.lo())
            }
            .unwrap()
        });

        initial_spans
    }

    /// Iterate through the sorted `CoverageSpan`s, and return the refined list of merged and
    /// de-duplicated `CoverageSpan`s.
    fn to_refined_spans(mut self) -> Vec<CoverageSpan> {
        while self.next_coverage_span() {
            if self.some_prev.is_none() {
                debug!("  initial span");
                self.check_invoked_macro_name_span();
            } else if self.curr().is_mergeable(self.prev()) {
                debug!("  same bcb (and neither is a closure), merge with prev={:?}", self.prev());
                let prev = self.take_prev();
                self.curr_mut().merge_from(prev);
                self.check_invoked_macro_name_span();
            // Note that curr.span may now differ from curr_original_span
            } else if self.prev_ends_before_curr() {
                debug!(
                    "  different bcbs and disjoint spans, so keep curr for next iter, and add \
                    prev={:?}",
                    self.prev()
                );
                let prev = self.take_prev();
                self.push_refined_span(prev);
                self.check_invoked_macro_name_span();
            } else if self.prev().is_closure {
                // drop any equal or overlapping span (`curr`) and keep `prev` to test again in the
                // next iter
                debug!(
                    "  curr overlaps a closure (prev). Drop curr and keep prev for next iter. \
                    prev={:?}",
                    self.prev()
                );
                self.take_curr();
            } else if self.curr().is_closure {
                self.carve_out_span_for_closure();
            } else if self.prev_original_span == self.curr().span {
                // Note that this compares the new (`curr`) span to `prev_original_span`.
                // In this branch, the actual span byte range of `prev_original_span` is not
                // important. What is important is knowing whether the new `curr` span was
                // **originally** the same as the original span of `prev()`. The original spans
                // reflect their original sort order, and for equal spans, conveys a partial
                // ordering based on CFG dominator priority.
                if self.prev().is_macro_expansion() && self.curr().is_macro_expansion() {
                    // Macros that expand to include branching (such as
                    // `assert_eq!()`, `assert_ne!()`, `info!()`, `debug!()`, or
                    // `trace!()) typically generate callee spans with identical
                    // ranges (typically the full span of the macro) for all
                    // `BasicBlocks`. This makes it impossible to distinguish
                    // the condition (`if val1 != val2`) from the optional
                    // branched statements (such as the call to `panic!()` on
                    // assert failure). In this case it is better (or less
                    // worse) to drop the optional branch bcbs and keep the
                    // non-conditional statements, to count when reached.
                    debug!(
                        "  curr and prev are part of a macro expansion, and curr has the same span \
                        as prev, but is in a different bcb. Drop curr and keep prev for next iter. \
                        prev={:?}",
                        self.prev()
                    );
                    self.take_curr();
                } else {
                    self.hold_pending_dups_unless_dominated();
                }
            } else {
                self.cutoff_prev_at_overlapping_curr();
                self.check_invoked_macro_name_span();
            }
        }

        debug!("    AT END, adding last prev={:?}", self.prev());
        let prev = self.take_prev();
        let pending_dups = self.pending_dups.split_off(0);
        for dup in pending_dups {
            debug!("    ...adding at least one pending dup={:?}", dup);
            self.push_refined_span(dup);
        }

        // Async functions wrap a closure that implements the body to be executed. The enclosing
        // function is called and returns an `impl Future` without initially executing any of the
        // body. To avoid showing the return from the enclosing function as a "covered" return from
        // the closure, the enclosing function's `TerminatorKind::Return`s `CoverageSpan` is
        // excluded. The closure's `Return` is the only one that will be counted. This provides
        // adequate coverage, and more intuitive counts. (Avoids double-counting the closing brace
        // of the function body.)
        let body_ends_with_closure = if let Some(last_covspan) = self.refined_spans.last() {
            last_covspan.is_closure && last_covspan.span.hi() == self.body_span.hi()
        } else {
            false
        };

        if !body_ends_with_closure {
            self.push_refined_span(prev);
        }

        // Remove `CoverageSpan`s derived from closures, originally added to ensure the coverage
        // regions for the current function leave room for the closure's own coverage regions
        // (injected separately, from the closure's own MIR).
        self.refined_spans.retain(|covspan| !covspan.is_closure);
        self.refined_spans
    }

    fn push_refined_span(&mut self, covspan: CoverageSpan) {
        let len = self.refined_spans.len();
        if len > 0 {
            let last = &mut self.refined_spans[len - 1];
            if last.is_mergeable(&covspan) {
                debug!(
                    "merging new refined span with last refined span, last={:?}, covspan={:?}",
                    last, covspan
                );
                last.merge_from(covspan);
                return;
            }
        }
        self.refined_spans.push(covspan)
    }

    fn check_invoked_macro_name_span(&mut self) {
        if let Some(visible_macro) = self.curr().visible_macro(self.body_span) {
            if self.prev_expn_span.map_or(true, |prev_expn_span| {
                self.curr().expn_span.ctxt() != prev_expn_span.ctxt()
            }) {
                let merged_prefix_len = self.curr_original_span.lo() - self.curr().span.lo();
                let after_macro_bang =
                    merged_prefix_len + BytePos(visible_macro.as_str().len() as u32 + 1);
                let mut macro_name_cov = self.curr().clone();
                self.curr_mut().span =
                    self.curr().span.with_lo(self.curr().span.lo() + after_macro_bang);
                macro_name_cov.span =
                    macro_name_cov.span.with_hi(macro_name_cov.span.lo() + after_macro_bang);
                debug!(
                    "  and curr starts a new macro expansion, so add a new span just for \
                            the macro `{}!`, new span={:?}",
                    visible_macro, macro_name_cov
                );
                self.push_refined_span(macro_name_cov);
            }
        }
    }

    // Generate a set of `CoverageSpan`s from the filtered set of `Statement`s and `Terminator`s of
    // the `BasicBlock`(s) in the given `BasicCoverageBlockData`. One `CoverageSpan` is generated
    // for each `Statement` and `Terminator`. (Note that subsequent stages of coverage analysis will
    // merge some `CoverageSpan`s, at which point a `CoverageSpan` may represent multiple
    // `Statement`s and/or `Terminator`s.)
    fn bcb_to_initial_coverage_spans(
        &self,
        bcb: BasicCoverageBlock,
        bcb_data: &'a BasicCoverageBlockData,
    ) -> Vec<CoverageSpan> {
        bcb_data
            .basic_blocks
            .iter()
            .flat_map(|&bb| {
                let data = &self.mir_body[bb];
                data.statements
                    .iter()
                    .enumerate()
                    .filter_map(move |(index, statement)| {
                        filtered_statement_span(statement).map(|span| {
                            CoverageSpan::for_statement(
                                statement,
                                function_source_span(span, self.body_span),
                                span,
                                bcb,
                                bb,
                                index,
                            )
                        })
                    })
                    .chain(filtered_terminator_span(data.terminator()).map(|span| {
                        CoverageSpan::for_terminator(
                            function_source_span(span, self.body_span),
                            span,
                            bcb,
                            bb,
                        )
                    }))
            })
            .collect()
    }

    fn curr(&self) -> &CoverageSpan {
        self.some_curr
            .as_ref()
            .unwrap_or_else(|| bug!("invalid attempt to unwrap a None some_curr"))
    }

    fn curr_mut(&mut self) -> &mut CoverageSpan {
        self.some_curr
            .as_mut()
            .unwrap_or_else(|| bug!("invalid attempt to unwrap a None some_curr"))
    }

    fn prev(&self) -> &CoverageSpan {
        self.some_prev
            .as_ref()
            .unwrap_or_else(|| bug!("invalid attempt to unwrap a None some_prev"))
    }

    fn prev_mut(&mut self) -> &mut CoverageSpan {
        self.some_prev
            .as_mut()
            .unwrap_or_else(|| bug!("invalid attempt to unwrap a None some_prev"))
    }

    fn take_prev(&mut self) -> CoverageSpan {
        self.some_prev.take().unwrap_or_else(|| bug!("invalid attempt to unwrap a None some_prev"))
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
    fn check_pending_dups(&mut self) {
        if let Some(dup) = self.pending_dups.last() && dup.span != self.prev().span {
            debug!(
                "    SAME spans, but pending_dups are NOT THE SAME, so BCBs matched on \
                previous iteration, or prev started a new disjoint span"
            );
            if dup.span.hi() <= self.curr().span.lo() {
                let pending_dups = self.pending_dups.split_off(0);
                for dup in pending_dups.into_iter() {
                    debug!("    ...adding at least one pending={:?}", dup);
                    self.push_refined_span(dup);
                }
            } else {
                self.pending_dups.clear();
            }
        }
    }

    /// Advance `prev` to `curr` (if any), and `curr` to the next `CoverageSpan` in sorted order.
    fn next_coverage_span(&mut self) -> bool {
        if let Some(curr) = self.some_curr.take() {
            self.prev_expn_span = Some(curr.expn_span);
            self.some_prev = Some(curr);
            self.prev_original_span = self.curr_original_span;
        }
        while let Some(curr) = self.sorted_spans_iter.as_mut().unwrap().next() {
            debug!("FOR curr={:?}", curr);
            if self.some_prev.is_some() && self.prev_starts_after_next(&curr) {
                debug!(
                    "  prev.span starts after curr.span, so curr will be dropped (skipping past \
                    closure?); prev={:?}",
                    self.prev()
                );
            } else {
                // Save a copy of the original span for `curr` in case the `CoverageSpan` is changed
                // by `self.curr_mut().merge_from(prev)`.
                self.curr_original_span = curr.span;
                self.some_curr.replace(curr);
                self.check_pending_dups();
                return true;
            }
        }
        false
    }

    /// If called, then the next call to `next_coverage_span()` will *not* update `prev` with the
    /// `curr` coverage span.
    fn take_curr(&mut self) -> CoverageSpan {
        self.some_curr.take().unwrap_or_else(|| bug!("invalid attempt to unwrap a None some_curr"))
    }

    /// Returns true if the curr span should be skipped because prev has already advanced beyond the
    /// end of curr. This can only happen if a prior iteration updated `prev` to skip past a region
    /// of code, such as skipping past a closure.
    fn prev_starts_after_next(&self, next_curr: &CoverageSpan) -> bool {
        self.prev().span.lo() > next_curr.span.lo()
    }

    /// Returns true if the curr span starts past the end of the prev span, which means they don't
    /// overlap, so we now know the prev can be added to the refined coverage spans.
    fn prev_ends_before_curr(&self) -> bool {
        self.prev().span.hi() <= self.curr().span.lo()
    }

    /// If `prev`s span extends left of the closure (`curr`), carve out the closure's span from
    /// `prev`'s span. (The closure's coverage counters will be injected when processing the
    /// closure's own MIR.) Add the portion of the span to the left of the closure; and if the span
    /// extends to the right of the closure, update `prev` to that portion of the span. For any
    /// `pending_dups`, repeat the same process.
    fn carve_out_span_for_closure(&mut self) {
        let curr_span = self.curr().span;
        let left_cutoff = curr_span.lo();
        let right_cutoff = curr_span.hi();
        let has_pre_closure_span = self.prev().span.lo() < right_cutoff;
        let has_post_closure_span = self.prev().span.hi() > right_cutoff;
        let mut pending_dups = self.pending_dups.split_off(0);
        if has_pre_closure_span {
            let mut pre_closure = self.prev().clone();
            pre_closure.span = pre_closure.span.with_hi(left_cutoff);
            debug!("  prev overlaps a closure. Adding span for pre_closure={:?}", pre_closure);
            if !pending_dups.is_empty() {
                for mut dup in pending_dups.iter().cloned() {
                    dup.span = dup.span.with_hi(left_cutoff);
                    debug!("    ...and at least one pre_closure dup={:?}", dup);
                    self.push_refined_span(dup);
                }
            }
            self.push_refined_span(pre_closure);
        }
        if has_post_closure_span {
            // Mutate `prev.span()` to start after the closure (and discard curr).
            // (**NEVER** update `prev_original_span` because it affects the assumptions
            // about how the `CoverageSpan`s are ordered.)
            self.prev_mut().span = self.prev().span.with_lo(right_cutoff);
            debug!("  Mutated prev.span to start after the closure. prev={:?}", self.prev());
            for dup in pending_dups.iter_mut() {
                debug!("    ...and at least one overlapping dup={:?}", dup);
                dup.span = dup.span.with_lo(right_cutoff);
            }
            self.pending_dups.append(&mut pending_dups);
            let closure_covspan = self.take_curr();
            self.push_refined_span(closure_covspan); // since self.prev() was already updated
        } else {
            pending_dups.clear();
        }
    }

    /// Called if `curr.span` equals `prev_original_span` (and potentially equal to all
    /// `pending_dups` spans, if any). Keep in mind, `prev.span()` may have been changed.
    /// If prev.span() was merged into other spans (with matching BCB, for instance),
    /// `prev.span.hi()` will be greater than (further right of) `prev_original_span.hi()`.
    /// If prev.span() was split off to the right of a closure, prev.span().lo() will be
    /// greater than prev_original_span.lo(). The actual span of `prev_original_span` is
    /// not as important as knowing that `prev()` **used to have the same span** as `curr(),
    /// which means their sort order is still meaningful for determining the dominator
    /// relationship.
    ///
    /// When two `CoverageSpan`s have the same `Span`, dominated spans can be discarded; but if
    /// neither `CoverageSpan` dominates the other, both (or possibly more than two) are held,
    /// until their disposition is determined. In this latter case, the `prev` dup is moved into
    /// `pending_dups` so the new `curr` dup can be moved to `prev` for the next iteration.
    fn hold_pending_dups_unless_dominated(&mut self) {
        // Equal coverage spans are ordered by dominators before dominated (if any), so it should be
        // impossible for `curr` to dominate any previous `CoverageSpan`.
        debug_assert!(!self.span_bcb_dominates(self.curr(), self.prev()));

        let initial_pending_count = self.pending_dups.len();
        if initial_pending_count > 0 {
            let mut pending_dups = self.pending_dups.split_off(0);
            pending_dups.retain(|dup| !self.span_bcb_dominates(dup, self.curr()));
            self.pending_dups.append(&mut pending_dups);
            if self.pending_dups.len() < initial_pending_count {
                debug!(
                    "  discarded {} of {} pending_dups that dominated curr",
                    initial_pending_count - self.pending_dups.len(),
                    initial_pending_count
                );
            }
        }

        if self.span_bcb_dominates(self.prev(), self.curr()) {
            debug!(
                "  different bcbs but SAME spans, and prev dominates curr. Discard prev={:?}",
                self.prev()
            );
            self.cutoff_prev_at_overlapping_curr();
        // If one span dominates the other, associate the span with the code from the dominated
        // block only (`curr`), and discard the overlapping portion of the `prev` span. (Note
        // that if `prev.span` is wider than `prev_original_span`, a `CoverageSpan` will still
        // be created for `prev`s block, for the non-overlapping portion, left of `curr.span`.)
        //
        // For example:
        //     match somenum {
        //         x if x < 1 => { ... }
        //     }...
        //
        // The span for the first `x` is referenced by both the pattern block (every time it is
        // evaluated) and the arm code (only when matched). The counter will be applied only to
        // the dominated block. This allows coverage to track and highlight things like the
        // assignment of `x` above, if the branch is matched, making `x` available to the arm
        // code; and to track and highlight the question mark `?` "try" operator at the end of
        // a function call returning a `Result`, so the `?` is covered when the function returns
        // an `Err`, and not counted as covered if the function always returns `Ok`.
        } else {
            // Save `prev` in `pending_dups`. (`curr` will become `prev` in the next iteration.)
            // If the `curr` CoverageSpan is later discarded, `pending_dups` can be discarded as
            // well; but if `curr` is added to refined_spans, the `pending_dups` will also be added.
            debug!(
                "  different bcbs but SAME spans, and neither dominates, so keep curr for \
                next iter, and, pending upcoming spans (unless overlapping) add prev={:?}",
                self.prev()
            );
            let prev = self.take_prev();
            self.pending_dups.push(prev);
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
            if self.prev().coverage_statements.is_empty() {
                debug!("  ... no non-overlapping statements to add");
            } else {
                debug!("  ... adding modified prev={:?}", self.prev());
                let prev = self.take_prev();
                self.push_refined_span(prev);
            }
        } else {
            // with `pending_dups`, `prev` cannot have any statements that don't overlap
            self.pending_dups.clear();
        }
    }

    fn span_bcb_dominates(&self, dom_covspan: &CoverageSpan, covspan: &CoverageSpan) -> bool {
        self.basic_coverage_blocks.dominates(dom_covspan.bcb, covspan.bcb)
    }
}

/// If the MIR `Statement` has a span contributive to computing coverage spans,
/// return it; otherwise return `None`.
pub(super) fn filtered_statement_span(statement: &Statement<'_>) -> Option<Span> {
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
        StatementKind::FakeRead(box (cause, _)) if cause == FakeReadCause::ForGuardBinding => None,

        // Retain spans from all other statements
        StatementKind::FakeRead(box (_, _)) // Not including `ForGuardBinding`
        | StatementKind::Intrinsic(..)
        | StatementKind::Assign(_)
        | StatementKind::SetDiscriminant { .. }
        | StatementKind::Deinit(..)
        | StatementKind::Retag(_, _)
        | StatementKind::AscribeUserType(_, _) => {
            Some(statement.source_info.span)
        }
    }
}

/// If the MIR `Terminator` has a span contributive to computing coverage spans,
/// return it; otherwise return `None`.
pub(super) fn filtered_terminator_span(terminator: &Terminator<'_>) -> Option<Span> {
    match terminator.kind {
        // These terminators have spans that don't positively contribute to computing a reasonable
        // span of actually executed source code. (For example, SwitchInt terminators extracted from
        // an `if condition { block }` has a span that includes the executed block, if true,
        // but for coverage, the code region executed, up to *and* through the SwitchInt,
        // actually stops before the if's block.)
        TerminatorKind::Unreachable // Unreachable blocks are not connected to the MIR CFG
        | TerminatorKind::Assert { .. }
        | TerminatorKind::Drop { .. }
        | TerminatorKind::DropAndReplace { .. }
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
        TerminatorKind::Resume
        | TerminatorKind::Abort
        | TerminatorKind::Return
        | TerminatorKind::Yield { .. }
        | TerminatorKind::GeneratorDrop
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
pub(super) fn function_source_span(span: Span, body_span: Span) -> Span {
    let original_span = original_sp(span, body_span).with_ctxt(body_span.ctxt());
    if body_span.contains(original_span) { original_span } else { body_span }
}
