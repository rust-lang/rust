// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Testing candidates
//
// After candidates have been simplified, the only match pairs that
// remain are those that require some sort of test. The functions here
// identify what tests are needed, perform the tests, and then filter
// the candidates based on the result.

use build::Builder;
use build::matches::{Candidate, MatchPair, Test, TestKind};
use hair::*;
use repr::*;

impl<H:Hair> Builder<H> {
    /// Identifies what test is needed to decide if `match_pair` is applicable.
    ///
    /// It is a bug to call this with a simplifyable pattern.
    pub fn test(&mut self, match_pair: &MatchPair<H>) -> Test<H> {
        match match_pair.pattern.kind.clone() {
            PatternKind::Variant { adt_def, variant_index, subpatterns } => {
                let elem = ProjectionElem::Downcast(adt_def, variant_index);
                let downcast_lvalue = match_pair.lvalue.clone().elem(elem);

                let consequent_match_pairs =
                    subpatterns.into_iter()
                               .map(|subpattern| {
                                   let lvalue =
                                       downcast_lvalue.clone().field(
                                           subpattern.field);
                                   self.match_pair(lvalue, subpattern.pattern)
                               })
                               .collect();

                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Switch { adt_def: adt_def },
                    outcome: variant_index,
                    match_pairs: consequent_match_pairs,
                }
            }

            PatternKind::Constant { expr } => {
                let expr = self.as_constant(expr);
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Eq { value: expr,
                                         ty: match_pair.pattern.ty.clone() },
                    outcome: 0, // 0 == true, of course. :)
                    match_pairs: vec![]
                }
            }

            PatternKind::Range { lo, hi } => {
                let lo = self.as_constant(lo);
                let hi = self.as_constant(hi);
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Range { lo: lo,
                                            hi: hi,
                                            ty: match_pair.pattern.ty.clone() },
                    outcome: 0, // 0 == true, of course. :)
                    match_pairs: vec![]
                }
            }

            PatternKind::Slice { prefix, slice: None, suffix } => {
                let len = prefix.len() + suffix.len();
                let mut consequent_match_pairs = vec![];
                self.append_prefix_suffix_pairs(
                    &mut consequent_match_pairs, match_pair.lvalue.clone(), prefix, suffix);
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Len { len: len, op: BinOp::Eq },
                    outcome: 0, // 0 == true, of course. :)
                    match_pairs: consequent_match_pairs
                }
            }

            PatternKind::Slice { prefix: _, slice: Some(_), suffix: _ } => {
                self.hir.span_bug(
                    match_pair.pattern.span,
                    &format!("slice patterns not implemented in MIR"));
            }

            PatternKind::Array { .. } |
            PatternKind::Wild |
            PatternKind::Binding { .. } |
            PatternKind::Leaf { .. } |
            PatternKind::Deref { .. } => {
                self.error_simplifyable(match_pair)
            }
        }
    }

    /// Generates the code to perform a test.
    pub fn perform_test(&mut self,
                        block: BasicBlock,
                        lvalue: &Lvalue<H>,
                        test: &Test<H>)
                        -> Vec<BasicBlock> {
        match test.kind.clone() {
            TestKind::Switch { adt_def } => {
                let num_enum_variants = self.hir.num_variants(adt_def);
                let target_blocks: Vec<_> =
                    (0..num_enum_variants).map(|_| self.cfg.start_new_block())
                                          .collect();
                self.cfg.terminate(block, Terminator::Switch {
                    discr: lvalue.clone(),
                    targets: target_blocks.clone()
                });
                target_blocks
            }

            TestKind::Eq { value, ty } => {
                // call PartialEq::eq(discrim, constant)
                let constant = self.push_constant(block, test.span, ty.clone(), value);
                let item_ref = self.hir.partial_eq(ty);
                self.call_comparison_fn(block, test.span, item_ref, lvalue.clone(), constant)
            }

            TestKind::Range { lo, hi, ty } => {
                // Test `v` by computing `PartialOrd::le(lo, v) && PartialOrd::le(v, hi)`.
                let lo = self.push_constant(block, test.span, ty.clone(), lo);
                let hi = self.push_constant(block, test.span, ty.clone(), hi);
                let item_ref = self.hir.partial_le(ty);

                let lo_blocks =
                    self.call_comparison_fn(block, test.span, item_ref.clone(), lo, lvalue.clone());

                let hi_blocks =
                    self.call_comparison_fn(lo_blocks[0], test.span, item_ref, lvalue.clone(), hi);

                let failure = self.cfg.start_new_block();
                self.cfg.terminate(lo_blocks[1], Terminator::Goto { target: failure });
                self.cfg.terminate(hi_blocks[1], Terminator::Goto { target: failure });

                vec![hi_blocks[0], failure]
            }

            TestKind::Len { len, op } => {
                let (usize_ty, bool_ty) = (self.hir.usize_ty(), self.hir.bool_ty());
                let (actual, result) = (self.temp(usize_ty), self.temp(bool_ty));

                // actual = len(lvalue)
                self.cfg.push_assign(
                    block, test.span,
                    &actual, Rvalue::Len(lvalue.clone()));

                // expected = <N>
                let expected =
                    self.push_usize(block, test.span, len);

                // result = actual == expected OR result = actual < expected
                self.cfg.push_assign(
                    block, test.span,
                    &result, Rvalue::BinaryOp(op,
                                              Operand::Consume(actual),
                                              Operand::Consume(expected)));

                // branch based on result
                let target_blocks: Vec<_> = vec![self.cfg.start_new_block(),
                                                 self.cfg.start_new_block()];
                self.cfg.terminate(block, Terminator::If {
                    cond: Operand::Consume(result),
                    targets: [target_blocks[0], target_blocks[1]]
                });

                target_blocks
            }
        }
    }

    fn call_comparison_fn(&mut self,
                          block: BasicBlock,
                          span: H::Span,
                          item_ref: ItemRef<H>,
                          lvalue1: Lvalue<H>,
                          lvalue2: Lvalue<H>)
                          -> Vec<BasicBlock> {
        let target_blocks = vec![self.cfg.start_new_block(),
                                 self.cfg.start_new_block()];

        let bool_ty = self.hir.bool_ty();
        let eq_result = self.temp(bool_ty);
        let func = self.push_item_ref(block, span, item_ref);
        let call_blocks = [self.cfg.start_new_block(), self.diverge_cleanup()];
        self.cfg.terminate(block,
                           Terminator::Call {
                               data: CallData {
                                   destination: eq_result.clone(),
                                   func: func,
                                   args: vec![lvalue1, lvalue2],
                               },
                               targets: call_blocks,
                           });

        // check the result
        self.cfg.terminate(call_blocks[0],
                           Terminator::If {
                               cond: Operand::Consume(eq_result),
                               targets: [target_blocks[0], target_blocks[1]]
                           });

        target_blocks
    }

    /// Given a candidate and the outcome of a test we have performed,
    /// transforms the candidate into a new candidate that reflects
    /// further tests still needed. Returns `None` if this candidate
    /// has now been ruled out.
    ///
    /// For example, if a candidate included the patterns `[x.0 @
    /// Ok(P1), x.1 @ 22]`, and we did a switch test on `x.0` and
    /// found the variant `Err` (as indicated by the `test_outcome`
    /// parameter), we would return `None`. But if the test_outcome
    /// were `Ok`, we would return `Some([x.0.downcast<Ok>.0 @ P1, x.1
    /// @ 22])`.
    pub fn candidate_under_assumption(&mut self,
                                      test_lvalue: &Lvalue<H>,
                                      test_kind: &TestKind<H>,
                                      test_outcome: usize,
                                      candidate: &Candidate<H>)
                                      -> Option<Candidate<H>> {
        let candidate = candidate.clone();
        let match_pairs = candidate.match_pairs;
        match self.match_pairs_under_assumption(test_lvalue, test_kind, test_outcome, match_pairs) {
            Some(match_pairs) => Some(Candidate { match_pairs: match_pairs, ..candidate }),
            None => None
        }
    }

    /// Helper for candidate_under_assumption that does the actual
    /// work of transforming the list of match pairs.
    fn match_pairs_under_assumption(&mut self,
                                    test_lvalue: &Lvalue<H>,
                                    test_kind: &TestKind<H>,
                                    test_outcome: usize,
                                    match_pairs: Vec<MatchPair<H>>)
                                    -> Option<Vec<MatchPair<H>>> {
        let mut result = vec![];
        for match_pair in match_pairs {
            // if the match pair is testing a different lvalue, it
            // is unaffected by this test.
            if match_pair.lvalue != *test_lvalue {
                result.push(match_pair);
                continue;
            }

            let desired_test = self.test(&match_pair);

            if *test_kind != desired_test.kind {
                // if the match pair wants to (e.g.) test for
                // equality against some particular constant, but
                // we did a switch, then we can't say whether it
                // matches or not, so we still have to include it
                // as a possibility.
                //
                // For example, we have a constant `FOO:
                // Option<i32> = Some(22)`, and `match_pair` is `x
                // @ FOO`, but we did a switch on the variant
                // (`Some` vs `None`). (OK, in principle this
                // could tell us something, but we're not that
                // smart yet to actually dig into the constant
                // itself)
                result.push(match_pair);
                continue;
            }

            if test_outcome != desired_test.outcome {
                // if we did the right kind of test, but it had the
                // wrong outcome, then this *entire candidate* can no
                // longer apply, huzzah! Therefore, we can stop this
                // iteration and just return `None` to our caller.
                return None;
            }

            // otherwise, the test passed, so we now have to include the
            // "unlocked" set of match pairs. For example, if we had `x @
            // Some(P1)`, and here we `test_kind==Switch` and
            // `outcome=Some`, then we would return `x.downcast<Some>.0 @
            // P1`.
            result.extend(desired_test.match_pairs);
        }
        Some(result)
    }

    fn error_simplifyable(&mut self, match_pair: &MatchPair<H>) -> ! {
        self.hir.span_bug(
            match_pair.pattern.span,
            &format!("simplifyable pattern found: {:?}", match_pair.pattern))
    }
}
