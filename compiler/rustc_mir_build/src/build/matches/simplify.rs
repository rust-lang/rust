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

use std::mem;

use tracing::{debug, instrument};

use crate::build::Builder;
use crate::build::matches::{MatchPairTree, PatternExtraData, TestCase};

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Simplify a list of match pairs so they all require a test. Stores relevant bindings and
    /// ascriptions in `extra_data`.
    #[instrument(skip(self), level = "debug")]
    pub(super) fn simplify_match_pairs<'pat>(
        &mut self,
        match_pairs: &mut Vec<MatchPairTree<'pat, 'tcx>>,
        extra_data: &mut PatternExtraData<'tcx>,
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
        // We therefore lower bindings from left-to-right, except we lower the `x` in `x @ pat`
        // after any bindings in `pat`. This doesn't work for or-patterns: the current structure of
        // match lowering forces us to lower bindings inside or-patterns last.
        for mut match_pair in mem::take(match_pairs) {
            self.simplify_match_pairs(&mut match_pair.subpairs, extra_data);
            if let TestCase::Irrefutable { binding, ascription } = match_pair.test_case {
                if let Some(binding) = binding {
                    extra_data.bindings.push(binding);
                }
                if let Some(ascription) = ascription {
                    extra_data.ascriptions.push(ascription);
                }
                // Simplifiable pattern; we replace it with its already simplified subpairs.
                match_pairs.append(&mut match_pair.subpairs);
            } else {
                // Unsimplifiable pattern; we keep it.
                match_pairs.push(match_pair);
            }
        }

        // Move or-patterns to the end, because they can result in us
        // creating additional candidates, so we want to test them as
        // late as possible.
        match_pairs.sort_by_key(|pair| matches!(pair.test_case, TestCase::Or { .. }));
        debug!(simplified = ?match_pairs, "simplify_match_pairs");
    }
}
