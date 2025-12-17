use std::cmp::Ordering;

use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::mir::{BinOp, Place};
use rustc_middle::span_bug;
use tracing::debug;

use crate::builder::Builder;
use crate::builder::matches::test::is_switch_ty;
use crate::builder::matches::{Candidate, Test, TestBranch, TestKind, TestableCase};

/// Output of [`Builder::partition_candidates_into_buckets`].
pub(crate) struct PartitionedCandidates<'tcx, 'b, 'c> {
    /// For each possible outcome of the test, the candidates that are matched in that outcome.
    pub(crate) target_candidates: FxIndexMap<TestBranch<'tcx>, Vec<&'b mut Candidate<'tcx>>>,
    /// The remaining candidates that weren't associated with any test outcome.
    pub(crate) remaining_candidates: &'b mut [&'c mut Candidate<'tcx>],
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Given a test, we partition the input candidates into several buckets.
    /// If a candidate matches in exactly one of the branches of `test`
    /// (and no other branches), we put it into the corresponding bucket.
    /// If it could match in more than one of the branches of `test`, the test
    /// doesn't usefully apply to it, and we stop partitioning candidates.
    ///
    /// Importantly, we also **mutate** the branched candidates to remove match pairs
    /// that are entailed by the outcome of the test, and add any sub-pairs of the
    /// removed pairs.
    ///
    /// For example:
    /// ```
    /// # let (x, y, z) = (true, true, true);
    /// match (x, y, z) {
    ///     (true , _    , true ) => true,  // (0)
    ///     (false, false, _    ) => false, // (1)
    ///     (_    , true , _    ) => true,  // (2)
    ///     (true , _    , false) => false, // (3)
    /// }
    /// # ;
    /// ```
    ///
    /// Assume we are testing on `x`. Conceptually, there are 2 overlapping candidate sets:
    /// - If the outcome is that `x` is true, candidates {0, 2, 3} are possible
    /// - If the outcome is that `x` is false, candidates {1, 2} are possible
    ///
    /// Following our algorithm:
    /// - Candidate 0 is bucketed into outcome `x == true`
    /// - Candidate 1 is bucketed into outcome `x == false`
    /// - Candidate 2 remains unbucketed, because testing `x` has no effect on it
    /// - Candidate 3 remains unbucketed, because a previous candidate (2) was unbucketed
    ///   - This helps preserve the illusion that candidates are tested "in order"
    ///
    /// The bucketed candidates are mutated to remove entailed match pairs:
    /// - candidate 0 becomes `[z @ true]` since we know that `x` was `true`;
    /// - candidate 1 becomes `[y @ false]` since we know that `x` was `false`.
    pub(super) fn partition_candidates_into_buckets<'b, 'c>(
        &mut self,
        match_place: Place<'tcx>,
        test: &Test<'tcx>,
        mut candidates: &'b mut [&'c mut Candidate<'tcx>],
    ) -> PartitionedCandidates<'tcx, 'b, 'c> {
        // For each of the possible outcomes, collect a vector of candidates that apply if the test
        // has that particular outcome.
        let mut target_candidates: FxIndexMap<_, Vec<&mut Candidate<'_>>> = Default::default();

        let total_candidate_count = candidates.len();

        // Partition the candidates into the appropriate vector in `target_candidates`.
        // Note that at some point we may encounter a candidate where the test is not relevant;
        // at that point, we stop partitioning.
        while let Some(candidate) = candidates.first_mut() {
            let Some(branch) =
                self.choose_bucket_for_candidate(match_place, test, candidate, &target_candidates)
            else {
                break;
            };
            let (candidate, rest) = candidates.split_first_mut().unwrap();
            target_candidates.entry(branch).or_insert_with(Vec::new).push(candidate);
            candidates = rest;
        }

        // At least the first candidate ought to be tested
        assert!(
            total_candidate_count > candidates.len(),
            "{total_candidate_count}, {candidates:#?}"
        );
        debug!("tested_candidates: {}", total_candidate_count - candidates.len());
        debug!("untested_candidates: {}", candidates.len());

        PartitionedCandidates { target_candidates, remaining_candidates: candidates }
    }

    /// Given that we are performing `test` against `test_place`, this job
    /// sorts out what the status of `candidate` will be after the test. See
    /// `test_candidates` for the usage of this function. The candidate may
    /// be modified to update its `match_pairs`.
    ///
    /// So, for example, if this candidate is `x @ Some(P0)` and the `Test` is
    /// a variant test, then we would modify the candidate to be `(x as
    /// Option).0 @ P0` and return the index corresponding to the variant
    /// `Some`.
    ///
    /// However, in some cases, the test may just not be relevant to candidate.
    /// For example, suppose we are testing whether `foo.x == 22`, but in one
    /// match arm we have `Foo { x: _, ... }`... in that case, the test for
    /// the value of `x` has no particular relevance to this candidate. In
    /// such cases, this function just returns None without doing anything.
    /// This is used by the overall `match_candidates` algorithm to structure
    /// the match as a whole. See `match_candidates` for more details.
    ///
    /// FIXME(#29623). In some cases, we have some tricky choices to make. for
    /// example, if we are testing that `x == 22`, but the candidate is `x @
    /// 13..55`, what should we do? In the event that the test is true, we know
    /// that the candidate applies, but in the event of false, we don't know
    /// that it *doesn't* apply. For now, we return false, indicate that the
    /// test does not apply to this candidate, but it might be we can get
    /// tighter match code if we do something a bit different.
    fn choose_bucket_for_candidate(
        &mut self,
        test_place: Place<'tcx>,
        test: &Test<'tcx>,
        candidate: &mut Candidate<'tcx>,
        // Other candidates that have already been partitioned into a bucket for this test, if any
        prior_candidates: &FxIndexMap<TestBranch<'tcx>, Vec<&mut Candidate<'tcx>>>,
    ) -> Option<TestBranch<'tcx>> {
        // Find the match_pair for this place (if any). At present,
        // afaik, there can be at most one. (In the future, if we
        // adopted a more general `@` operator, there might be more
        // than one, but it'd be very unusual to have two sides that
        // both require tests; you'd expect one side to be simplified
        // away.)
        let (match_pair_index, match_pair) = candidate
            .match_pairs
            .iter()
            .enumerate()
            .find(|&(_, mp)| mp.place == Some(test_place))?;

        // If true, the match pair is completely entailed by its corresponding test
        // branch, so it can be removed. If false, the match pair is _compatible_
        // with its test branch, but still needs a more specific test.
        let fully_matched;
        let ret = match (&test.kind, &match_pair.testable_case) {
            // If we are performing a variant switch, then this
            // informs variant patterns, but nothing else.
            (
                &TestKind::Switch { adt_def: tested_adt_def },
                &TestableCase::Variant { adt_def, variant_index },
            ) => {
                assert_eq!(adt_def, tested_adt_def);
                fully_matched = true;
                Some(TestBranch::Variant(variant_index))
            }

            // If we are performing a switch over integers, then this informs integer
            // equality, but nothing else.
            //
            // FIXME(#29623) we could use PatKind::Range to rule
            // things out here, in some cases.
            //
            // FIXME(Zalathar): Is the `is_switch_ty` test unnecessary?
            (TestKind::SwitchInt, &TestableCase::Constant { value })
                if is_switch_ty(match_pair.pattern_ty) =>
            {
                // An important invariant of candidate bucketing is that a candidate
                // must not match in multiple branches. For `SwitchInt` tests, adding
                // a new value might invalidate that property for range patterns that
                // have already been partitioned into the failure arm, so we must take care
                // not to add such values here.
                let is_covering_range = |testable_case: &TestableCase<'tcx>| {
                    testable_case.as_range().is_some_and(|range| {
                        matches!(range.contains(value, self.tcx), None | Some(true))
                    })
                };
                let is_conflicting_candidate = |candidate: &&mut Candidate<'tcx>| {
                    candidate.match_pairs.iter().any(|mp| {
                        mp.place == Some(test_place) && is_covering_range(&mp.testable_case)
                    })
                };
                if prior_candidates
                    .get(&TestBranch::Failure)
                    .is_some_and(|candidates| candidates.iter().any(is_conflicting_candidate))
                {
                    fully_matched = false;
                    None
                } else {
                    fully_matched = true;
                    Some(TestBranch::Constant(value))
                }
            }
            (TestKind::SwitchInt, TestableCase::Range(range)) => {
                // When performing a `SwitchInt` test, a range pattern can be
                // sorted into the failure arm if it doesn't contain _any_ of
                // the values being tested. (This restricts what values can be
                // added to the test by subsequent candidates.)
                fully_matched = false;
                let not_contained = prior_candidates
                    .keys()
                    .filter_map(|br| br.as_constant())
                    .all(|val| matches!(range.contains(val, self.tcx), Some(false)));

                not_contained.then(|| {
                    // No switch values are contained in the pattern range,
                    // so the pattern can be matched only if this test fails.
                    TestBranch::Failure
                })
            }

            (TestKind::If, TestableCase::Constant { value }) => {
                fully_matched = true;
                let value = value.try_to_bool().unwrap_or_else(|| {
                    span_bug!(test.span, "expected boolean value but got {value:?}")
                });
                Some(if value { TestBranch::Success } else { TestBranch::Failure })
            }

            (
                &TestKind::Len { len: test_len, op: BinOp::Eq },
                &TestableCase::Slice { len, variable_length },
            ) => {
                match (test_len.cmp(&(len as u64)), variable_length) {
                    (Ordering::Equal, false) => {
                        // on true, min_len = len = $actual_length,
                        // on false, len != $actual_length
                        fully_matched = true;
                        Some(TestBranch::Success)
                    }
                    (Ordering::Less, _) => {
                        // test_len < pat_len. If $actual_len = test_len,
                        // then $actual_len < pat_len and we don't have
                        // enough elements.
                        fully_matched = false;
                        Some(TestBranch::Failure)
                    }
                    (Ordering::Equal | Ordering::Greater, true) => {
                        // This can match both if $actual_len = test_len >= pat_len,
                        // and if $actual_len > test_len. We can't advance.
                        fully_matched = false;
                        None
                    }
                    (Ordering::Greater, false) => {
                        // test_len != pat_len, so if $actual_len = test_len, then
                        // $actual_len != pat_len.
                        fully_matched = false;
                        Some(TestBranch::Failure)
                    }
                }
            }
            (
                &TestKind::Len { len: test_len, op: BinOp::Ge },
                &TestableCase::Slice { len, variable_length },
            ) => {
                // the test is `$actual_len >= test_len`
                match (test_len.cmp(&(len as u64)), variable_length) {
                    (Ordering::Equal, true) => {
                        // $actual_len >= test_len = pat_len,
                        // so we can match.
                        fully_matched = true;
                        Some(TestBranch::Success)
                    }
                    (Ordering::Less, _) | (Ordering::Equal, false) => {
                        // test_len <= pat_len. If $actual_len < test_len,
                        // then it is also < pat_len, so the test passing is
                        // necessary (but insufficient).
                        fully_matched = false;
                        Some(TestBranch::Success)
                    }
                    (Ordering::Greater, false) => {
                        // test_len > pat_len. If $actual_len >= test_len > pat_len,
                        // then we know we won't have a match.
                        fully_matched = false;
                        Some(TestBranch::Failure)
                    }
                    (Ordering::Greater, true) => {
                        // test_len < pat_len, and is therefore less
                        // strict. This can still go both ways.
                        fully_matched = false;
                        None
                    }
                }
            }

            (TestKind::Range(test), TestableCase::Range(pat)) => {
                if test == pat {
                    fully_matched = true;
                    Some(TestBranch::Success)
                } else {
                    fully_matched = false;
                    // If the testing range does not overlap with pattern range,
                    // the pattern can be matched only if this test fails.
                    if !test.overlaps(pat, self.tcx)? { Some(TestBranch::Failure) } else { None }
                }
            }
            (TestKind::Range(range), &TestableCase::Constant { value }) => {
                fully_matched = false;
                if !range.contains(value, self.tcx)? {
                    // `value` is not contained in the testing range,
                    // so `value` can be matched only if this test fails.
                    Some(TestBranch::Failure)
                } else {
                    None
                }
            }

            (TestKind::Eq { value: test_val, .. }, TestableCase::Constant { value: case_val }) => {
                if test_val == case_val {
                    fully_matched = true;
                    Some(TestBranch::Success)
                } else {
                    fully_matched = false;
                    Some(TestBranch::Failure)
                }
            }

            (TestKind::Deref { temp: test_temp, .. }, TestableCase::Deref { temp, .. })
                if test_temp == temp =>
            {
                fully_matched = true;
                Some(TestBranch::Success)
            }

            (TestKind::Never, _) => {
                fully_matched = true;
                Some(TestBranch::Success)
            }

            (
                TestKind::Switch { .. }
                | TestKind::SwitchInt { .. }
                | TestKind::If
                | TestKind::Len { .. }
                | TestKind::Range { .. }
                | TestKind::Eq { .. }
                | TestKind::Deref { .. },
                _,
            ) => {
                fully_matched = false;
                None
            }
        };

        if fully_matched {
            // Replace the match pair by its sub-pairs.
            let match_pair = candidate.match_pairs.remove(match_pair_index);
            candidate.match_pairs.extend(match_pair.subpairs);
            // Move or-patterns to the end.
            candidate.sort_match_pairs();
        }

        ret
    }
}
