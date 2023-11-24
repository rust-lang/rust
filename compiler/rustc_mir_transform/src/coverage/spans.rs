use std::cell::OnceCell;

use rustc_data_structures::graph::WithNumNodes;
use rustc_index::IndexVec;
use rustc_middle::mir;
use rustc_span::{BytePos, ExpnKind, MacroKind, Span, Symbol, DUMMY_SP};

use super::graph::{BasicCoverageBlock, CoverageGraph, START_BCB};

mod from_mir;

pub(super) struct CoverageSpans {
    /// Map from BCBs to their list of coverage spans.
    bcb_to_spans: IndexVec<BasicCoverageBlock, Vec<Span>>,
}

impl CoverageSpans {
    pub(super) fn generate_coverage_spans(
        mir_body: &mir::Body<'_>,
        fn_sig_span: Span,
        body_span: Span,
        basic_coverage_blocks: &CoverageGraph,
    ) -> Self {
        let coverage_spans = CoverageSpansGenerator::generate_coverage_spans(
            mir_body,
            fn_sig_span,
            body_span,
            basic_coverage_blocks,
        );

        // Group the coverage spans by BCB, with the BCBs in sorted order.
        let mut bcb_to_spans = IndexVec::from_elem_n(Vec::new(), basic_coverage_blocks.num_nodes());
        for CoverageSpan { bcb, span, .. } in coverage_spans {
            bcb_to_spans[bcb].push(span);
        }

        Self { bcb_to_spans }
    }

    pub(super) fn bcb_has_coverage_spans(&self, bcb: BasicCoverageBlock) -> bool {
        !self.bcb_to_spans[bcb].is_empty()
    }

    pub(super) fn spans_for_bcb(&self, bcb: BasicCoverageBlock) -> &[Span] {
        &self.bcb_to_spans[bcb]
    }
}

/// A BCB is deconstructed into one or more `Span`s. Each `Span` maps to a `CoverageSpan` that
/// references the originating BCB and one or more MIR `Statement`s and/or `Terminator`s.
/// Initially, the `Span`s come from the `Statement`s and `Terminator`s, but subsequent
/// transforms can combine adjacent `Span`s and `CoverageSpan` from the same BCB, merging the
/// `merged_spans` vectors, and the `Span`s to cover the extent of the combined `Span`s.
///
/// Note: A span merged into another CoverageSpan may come from a `BasicBlock` that
/// is not part of the `CoverageSpan` bcb if the statement was included because it's `Span` matches
/// or is subsumed by the `Span` associated with this `CoverageSpan`, and it's `BasicBlock`
/// `dominates()` the `BasicBlock`s in this `CoverageSpan`.
#[derive(Debug, Clone)]
struct CoverageSpan {
    pub span: Span,
    pub expn_span: Span,
    pub current_macro_or_none: OnceCell<Option<Symbol>>,
    pub bcb: BasicCoverageBlock,
    /// List of all the original spans from MIR that have been merged into this
    /// span. Mainly used to precisely skip over gaps when truncating a span.
    pub merged_spans: Vec<Span>,
    pub is_closure: bool,
}

impl CoverageSpan {
    pub fn for_fn_sig(fn_sig_span: Span) -> Self {
        Self::new(fn_sig_span, fn_sig_span, START_BCB, false)
    }

    pub(super) fn new(
        span: Span,
        expn_span: Span,
        bcb: BasicCoverageBlock,
        is_closure: bool,
    ) -> Self {
        Self {
            span,
            expn_span,
            current_macro_or_none: Default::default(),
            bcb,
            merged_spans: vec![span],
            is_closure,
        }
    }

    pub fn merge_from(&mut self, mut other: CoverageSpan) {
        debug_assert!(self.is_mergeable(&other));
        self.span = self.span.to(other.span);
        self.merged_spans.append(&mut other.merged_spans);
    }

    pub fn cutoff_statements_at(&mut self, cutoff_pos: BytePos) {
        self.merged_spans.retain(|span| span.hi() <= cutoff_pos);
        if let Some(max_hi) = self.merged_spans.iter().map(|span| span.hi()).max() {
            self.span = self.span.with_hi(max_hi);
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

    /// If the span is part of a macro, returns the macro name symbol.
    pub fn current_macro(&self) -> Option<Symbol> {
        self.current_macro_or_none
            .get_or_init(|| {
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
        if let Some(current_macro) = self.current_macro()
            && self
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
struct CoverageSpansGenerator<'a> {
    /// A `Span` covering the function body of the MIR (typically from left curly brace to right
    /// curly brace).
    body_span: Span,

    /// The BasicCoverageBlock Control Flow Graph (BCB CFG).
    basic_coverage_blocks: &'a CoverageGraph,

    /// The initial set of `CoverageSpan`s, sorted by `Span` (`lo` and `hi`) and by relative
    /// dominance between the `BasicCoverageBlock`s of equal `Span`s.
    sorted_spans_iter: std::vec::IntoIter<CoverageSpan>,

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

impl<'a> CoverageSpansGenerator<'a> {
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
        mir_body: &mir::Body<'_>,
        fn_sig_span: Span, // Ensured to be same SourceFile and SyntaxContext as `body_span`
        body_span: Span,
        basic_coverage_blocks: &'a CoverageGraph,
    ) -> Vec<CoverageSpan> {
        let sorted_spans = from_mir::mir_to_initial_sorted_coverage_spans(
            mir_body,
            fn_sig_span,
            body_span,
            basic_coverage_blocks,
        );

        let coverage_spans = Self {
            body_span,
            basic_coverage_blocks,
            sorted_spans_iter: sorted_spans.into_iter(),
            some_curr: None,
            curr_original_span: DUMMY_SP,
            some_prev: None,
            prev_original_span: DUMMY_SP,
            pending_dups: Vec::new(),
            refined_spans: Vec::with_capacity(basic_coverage_blocks.num_nodes() * 2),
        };

        coverage_spans.to_refined_spans()
    }

    /// Iterate through the sorted `CoverageSpan`s, and return the refined list of merged and
    /// de-duplicated `CoverageSpan`s.
    fn to_refined_spans(mut self) -> Vec<CoverageSpan> {
        while self.next_coverage_span() {
            // For the first span we don't have `prev` set, so most of the
            // span-processing steps don't make sense yet.
            if self.some_prev.is_none() {
                debug!("  initial span");
                self.maybe_push_macro_name_span();
                continue;
            }

            // The remaining cases assume that `prev` and `curr` are set.
            let prev = self.prev();
            let curr = self.curr();

            if curr.is_mergeable(prev) {
                debug!("  same bcb (and neither is a closure), merge with prev={prev:?}");
                let prev = self.take_prev();
                self.curr_mut().merge_from(prev);
                self.maybe_push_macro_name_span();
            // Note that curr.span may now differ from curr_original_span
            } else if prev.span.hi() <= curr.span.lo() {
                debug!(
                    "  different bcbs and disjoint spans, so keep curr for next iter, and add prev={prev:?}",
                );
                let prev = self.take_prev();
                self.push_refined_span(prev);
                self.maybe_push_macro_name_span();
            } else if prev.is_closure {
                // drop any equal or overlapping span (`curr`) and keep `prev` to test again in the
                // next iter
                debug!(
                    "  curr overlaps a closure (prev). Drop curr and keep prev for next iter. prev={prev:?}",
                );
                self.take_curr(); // Discards curr.
            } else if curr.is_closure {
                self.carve_out_span_for_closure();
            } else if self.prev_original_span == curr.span {
                // Note that this compares the new (`curr`) span to `prev_original_span`.
                // In this branch, the actual span byte range of `prev_original_span` is not
                // important. What is important is knowing whether the new `curr` span was
                // **originally** the same as the original span of `prev()`. The original spans
                // reflect their original sort order, and for equal spans, conveys a partial
                // ordering based on CFG dominator priority.
                if prev.is_macro_expansion() && curr.is_macro_expansion() {
                    // Macros that expand to include branching (such as
                    // `assert_eq!()`, `assert_ne!()`, `info!()`, `debug!()`, or
                    // `trace!()`) typically generate callee spans with identical
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
                        prev={prev:?}",
                    );
                    self.take_curr(); // Discards curr.
                } else {
                    self.update_pending_dups();
                }
            } else {
                self.cutoff_prev_at_overlapping_curr();
                self.maybe_push_macro_name_span();
            }
        }

        let prev = self.take_prev();
        debug!("    AT END, adding last prev={prev:?}");

        // Take `pending_dups` so that we can drain it while calling self methods.
        // It is never used as a field after this point.
        for dup in std::mem::take(&mut self.pending_dups) {
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
        if let Some(last) = self.refined_spans.last_mut()
            && last.is_mergeable(&covspan)
        {
            // Instead of pushing the new span, merge it with the last refined span.
            debug!(?last, ?covspan, "merging new refined span with last refined span");
            last.merge_from(covspan);
        } else {
            self.refined_spans.push(covspan);
        }
    }

    /// If `curr` is part of a new macro expansion, carve out and push a separate
    /// span that ends just after the macro name and its subsequent `!`.
    fn maybe_push_macro_name_span(&mut self) {
        let curr = self.curr();

        let Some(visible_macro) = curr.visible_macro(self.body_span) else { return };
        if let Some(prev) = &self.some_prev
            && prev.expn_span.eq_ctxt(curr.expn_span)
        {
            return;
        }

        let merged_prefix_len = self.curr_original_span.lo() - curr.span.lo();
        let after_macro_bang = merged_prefix_len + BytePos(visible_macro.as_str().len() as u32 + 1);
        if self.curr().span.lo() + after_macro_bang > self.curr().span.hi() {
            // Something is wrong with the macro name span;
            // return now to avoid emitting malformed mappings.
            // FIXME(#117788): Track down why this happens.
            return;
        }
        let mut macro_name_cov = curr.clone();
        self.curr_mut().span = curr.span.with_lo(curr.span.lo() + after_macro_bang);
        macro_name_cov.span =
            macro_name_cov.span.with_hi(macro_name_cov.span.lo() + after_macro_bang);
        debug!(
            "  and curr starts a new macro expansion, so add a new span just for \
            the macro `{visible_macro}!`, new span={macro_name_cov:?}",
        );
        self.push_refined_span(macro_name_cov);
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

    /// If called, then the next call to `next_coverage_span()` will *not* update `prev` with the
    /// `curr` coverage span.
    fn take_curr(&mut self) -> CoverageSpan {
        self.some_curr.take().unwrap_or_else(|| bug!("invalid attempt to unwrap a None some_curr"))
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
            // Temporarily steal `pending_dups` into a local, so that we can
            // drain it while calling other self methods.
            let mut pending_dups = std::mem::take(&mut self.pending_dups);
            for dup in pending_dups.drain(..) {
                debug!("    ...adding at least one pending={:?}", dup);
                self.push_refined_span(dup);
            }
            // The list of dups is now empty, but we can recycle its capacity.
            assert!(pending_dups.is_empty() && self.pending_dups.is_empty());
            self.pending_dups = pending_dups;
        } else {
            self.pending_dups.clear();
        }
    }

    /// Advance `prev` to `curr` (if any), and `curr` to the next `CoverageSpan` in sorted order.
    fn next_coverage_span(&mut self) -> bool {
        if let Some(curr) = self.some_curr.take() {
            self.some_prev = Some(curr);
            self.prev_original_span = self.curr_original_span;
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
                // Save a copy of the original span for `curr` in case the `CoverageSpan` is changed
                // by `self.curr_mut().merge_from(prev)`.
                self.curr_original_span = curr.span;
                self.some_curr.replace(curr);
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

        // Temporarily steal `pending_dups` into a local, so that we can
        // mutate and/or drain it while calling other self methods.
        let mut pending_dups = std::mem::take(&mut self.pending_dups);

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
            let closure_covspan = self.take_curr(); // Prevent this curr from becoming prev.
            self.push_refined_span(closure_covspan); // since self.prev() was already updated
        } else {
            pending_dups.clear();
        }

        // Restore the modified post-closure spans, or the empty vector's capacity.
        assert!(self.pending_dups.is_empty());
        self.pending_dups = pending_dups;
    }

    /// Called if `curr.span` equals `prev_original_span` (and potentially equal to all
    /// `pending_dups` spans, if any). Keep in mind, `prev.span()` may have been changed.
    /// If prev.span() was merged into other spans (with matching BCB, for instance),
    /// `prev.span.hi()` will be greater than (further right of) `prev_original_span.hi()`.
    /// If prev.span() was split off to the right of a closure, prev.span().lo() will be
    /// greater than prev_original_span.lo(). The actual span of `prev_original_span` is
    /// not as important as knowing that `prev()` **used to have the same span** as `curr()`,
    /// which means their sort order is still meaningful for determining the dominator
    /// relationship.
    ///
    /// When two `CoverageSpan`s have the same `Span`, dominated spans can be discarded; but if
    /// neither `CoverageSpan` dominates the other, both (or possibly more than two) are held,
    /// until their disposition is determined. In this latter case, the `prev` dup is moved into
    /// `pending_dups` so the new `curr` dup can be moved to `prev` for the next iteration.
    fn update_pending_dups(&mut self) {
        let prev_bcb = self.prev().bcb;
        let curr_bcb = self.curr().bcb;

        // Equal coverage spans are ordered by dominators before dominated (if any), so it should be
        // impossible for `curr` to dominate any previous `CoverageSpan`.
        debug_assert!(!self.basic_coverage_blocks.dominates(curr_bcb, prev_bcb));

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

        if self.basic_coverage_blocks.dominates(prev_bcb, curr_bcb) {
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
            if self.prev().merged_spans.is_empty() {
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
}
