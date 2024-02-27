//! Simplifying Candidates
//!
//! *Simplifying* a match pair `place @ pattern` means breaking it down
//! into bindings or other, simpler match pairs. For example:
//!
//! - `place @ (P1, P2)` can be simplified to `[place.0 @ P1, place.1 @ P2]`
//! - `place @ x` can be simplified to `[]` by binding `x` to `place`
//!
//! The `simplify_match_pairs` routine just repeatedly applies these
//! sort of simplifications until there is nothing left to
//! simplify. Match pairs cannot be simplified if they require some
//! sort of test: for example, testing which variant an enum is, or
//! testing a value against a constant.

use crate::build::matches::{Ascription, Binding, Candidate, FlatPat, MatchPair, TestCase};
use crate::build::Builder;

use std::mem;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Simplify a list of match pairs so they all require a test. Stores relevant bindings and
    /// ascriptions in the provided `Vec`s.
    #[instrument(skip(self), level = "debug")]
    pub(super) fn simplify_match_pairs<'pat>(
        &mut self,
        match_pairs: &mut Vec<MatchPair<'pat, 'tcx>>,
        candidate_bindings: &mut Vec<Binding<'tcx>>,
        candidate_ascriptions: &mut Vec<Ascription<'tcx>>,
    ) {
        // In order to please the borrow checker, in a pattern like `x @ pat` we must lower the
        // bindings in `pat` before `x`. E.g. (#69971):
        //
        // struct NonCopyStruct {
        //     copy_field: u32,
        // }
        //
        // fn foo1(x: NonCopyStruct) {
        //     let y @ NonCopyStruct { copy_field: z } = x;
        //     // the above should turn into
        //     let z = x.copy_field;
        //     let y = x;
        // }
        //
        // We can't just reverse the binding order, because we must preserve pattern-order
        // otherwise, e.g. in `let (Some(a), Some(b)) = (x, y)`. Our rule then is: deepest-first,
        // and bindings at the same depth stay in source order.
        //
        // To do this, every time around the loop we prepend the newly found bindings to the
        // bindings we already had.
        //
        // example:
        // candidate.bindings = [1, 2, 3]
        // bindings in iter 1: [4, 5]
        // bindings in iter 2: [6, 7]
        //
        // final bindings: [6, 7, 4, 5, 1, 2, 3]
        let mut accumulated_bindings = mem::take(candidate_bindings);
        let mut simplified_match_pairs = Vec::new();
        // Repeatedly simplify match pairs until we're left with only unsimplifiable ones.
        loop {
            for mut match_pair in mem::take(match_pairs) {
                if let TestCase::Irrefutable { binding, ascription } = match_pair.test_case {
                    if let Some(binding) = binding {
                        candidate_bindings.push(binding);
                    }
                    if let Some(ascription) = ascription {
                        candidate_ascriptions.push(ascription);
                    }
                    // Simplifiable pattern; we replace it with its subpairs and simplify further.
                    match_pairs.append(&mut match_pair.subpairs);
                } else {
                    // Unsimplifiable pattern; we recursively simplify its subpairs and don't
                    // process it further.
                    self.simplify_match_pairs(
                        &mut match_pair.subpairs,
                        candidate_bindings,
                        candidate_ascriptions,
                    );
                    simplified_match_pairs.push(match_pair);
                }
            }

            // This does: accumulated_bindings = candidate.bindings.take() ++ accumulated_bindings
            candidate_bindings.extend_from_slice(&accumulated_bindings);
            mem::swap(candidate_bindings, &mut accumulated_bindings);
            candidate_bindings.clear();

            if match_pairs.is_empty() {
                break;
            }
        }

        // Store computed bindings back in `candidate_bindings`.
        mem::swap(candidate_bindings, &mut accumulated_bindings);
        // Store simplified match pairs back in `match_pairs`.
        mem::swap(match_pairs, &mut simplified_match_pairs);

        // Move or-patterns to the end, because they can result in us
        // creating additional candidates, so we want to test them as
        // late as possible.
        match_pairs.sort_by_key(|pair| matches!(pair.test_case, TestCase::Or { .. }));
        debug!(simplified = ?match_pairs, "simplify_match_pairs");
    }

    /// Create a new candidate for each pattern in `pats`, and recursively simplify tje
    /// single-or-pattern case.
    pub(super) fn create_or_subcandidates<'pat>(
        &mut self,
        pats: &[FlatPat<'pat, 'tcx>],
        has_guard: bool,
    ) -> Vec<Candidate<'pat, 'tcx>> {
        pats.iter()
            .cloned()
            .map(|flat_pat| {
                let mut candidate = Candidate::from_flat_pat(flat_pat, has_guard);
                if let [MatchPair { test_case: TestCase::Or { pats, .. }, .. }] =
                    &*candidate.match_pairs
                {
                    candidate.subcandidates = self.create_or_subcandidates(pats, has_guard);
                    candidate.match_pairs.pop();
                }
                candidate
            })
            .collect()
    }
}
