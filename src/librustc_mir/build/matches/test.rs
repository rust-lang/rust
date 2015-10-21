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

use build::{BlockAnd, Builder};
use build::matches::{Candidate, MatchPair, Test, TestKind};
use hair::*;
use repr::*;
use syntax::codemap::Span;

impl<'a,'tcx> Builder<'a,'tcx> {
    /// Identifies what test is needed to decide if `match_pair` is applicable.
    ///
    /// It is a bug to call this with a simplifyable pattern.
    pub fn test(&mut self, match_pair: &MatchPair<'tcx>) -> Test<'tcx> {
        match match_pair.pattern.kind {
            PatternKind::Variant { ref adt_def, variant_index: _, subpatterns: _ } => {
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Switch { adt_def: adt_def.clone() },
                }
            }

            PatternKind::Constant { ref value } => {
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Eq {
                        value: value.clone(),
                        ty: match_pair.pattern.ty.clone(),
                    },
                }
            }

            PatternKind::Range { ref lo, ref hi } => {
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Range {
                        lo: lo.clone(),
                        hi: hi.clone(),
                        ty: match_pair.pattern.ty.clone(),
                    },
                }
            }

            PatternKind::Slice { ref prefix, ref slice, ref suffix } => {
                let len = prefix.len() + suffix.len();
                let op = if slice.is_some() {
                    BinOp::Ge
                } else {
                    BinOp::Eq
                };
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Len { len: len, op: op },
                }
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
                        lvalue: &Lvalue<'tcx>,
                        test: &Test<'tcx>)
                        -> Vec<BasicBlock> {
        match test.kind.clone() {
            TestKind::Switch { adt_def } => {
                let num_enum_variants = self.hir.num_variants(adt_def);
                let target_blocks: Vec<_> =
                    (0..num_enum_variants).map(|_| self.cfg.start_new_block())
                                          .collect();
                self.cfg.terminate(block, Terminator::Switch {
                    discr: lvalue.clone(),
                    adt_def: adt_def,
                    targets: target_blocks.clone()
                });
                target_blocks
            }

            TestKind::Eq { value, ty } => {
                // call PartialEq::eq(discrim, constant)
                let constant = self.push_literal(block, test.span, ty.clone(), value);
                let item_ref = self.hir.partial_eq(ty);
                self.call_comparison_fn(block, test.span, item_ref, lvalue.clone(), constant)
            }

            TestKind::Range { lo, hi, ty } => {
                // Test `v` by computing `PartialOrd::le(lo, v) && PartialOrd::le(v, hi)`.
                let lo = self.push_literal(block, test.span, ty.clone(), lo);
                let hi = self.push_literal(block, test.span, ty.clone(), hi);
                let item_ref = self.hir.partial_le(ty);

                let lo_blocks = self.call_comparison_fn(block,
                                                        test.span,
                                                        item_ref.clone(),
                                                        lo,
                                                        lvalue.clone());

                let hi_blocks = self.call_comparison_fn(lo_blocks[0],
                                                        test.span,
                                                        item_ref,
                                                        lvalue.clone(),
                                                        hi);

                let failure = self.cfg.start_new_block();
                self.cfg.terminate(lo_blocks[1], Terminator::Goto { target: failure });
                self.cfg.terminate(hi_blocks[1], Terminator::Goto { target: failure });

                vec![hi_blocks[0], failure]
            }

            TestKind::Len { len, op } => {
                let (usize_ty, bool_ty) = (self.hir.usize_ty(), self.hir.bool_ty());
                let (actual, result) = (self.temp(usize_ty), self.temp(bool_ty));

                // actual = len(lvalue)
                self.cfg.push_assign(block, test.span, &actual, Rvalue::Len(lvalue.clone()));

                // expected = <N>
                let expected = self.push_usize(block, test.span, len);

                // result = actual == expected OR result = actual < expected
                self.cfg.push_assign(block,
                                     test.span,
                                     &result,
                                     Rvalue::BinaryOp(op,
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
                          span: Span,
                          item_ref: ItemRef<'tcx>,
                          lvalue1: Lvalue<'tcx>,
                          lvalue2: Lvalue<'tcx>)
                          -> Vec<BasicBlock> {
        let target_blocks = vec![self.cfg.start_new_block(), self.cfg.start_new_block()];

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
                               targets: [target_blocks[0], target_blocks[1]],
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
                                      mut block: BasicBlock,
                                      test_lvalue: &Lvalue<'tcx>,
                                      test_kind: &TestKind<'tcx>,
                                      test_outcome: usize,
                                      candidate: &Candidate<'tcx>)
                                      -> BlockAnd<Option<Candidate<'tcx>>> {
        let candidate = candidate.clone();
        let match_pairs = candidate.match_pairs;
        let result = unpack!(block = self.match_pairs_under_assumption(block,
                                                                       test_lvalue,
                                                                       test_kind,
                                                                       test_outcome,
                                                                       match_pairs));
        block.and(match result {
            Some(match_pairs) => Some(Candidate { match_pairs: match_pairs, ..candidate }),
            None => None,
        })
    }

    /// Helper for candidate_under_assumption that does the actual
    /// work of transforming the list of match pairs.
    fn match_pairs_under_assumption(&mut self,
                                    mut block: BasicBlock,
                                    test_lvalue: &Lvalue<'tcx>,
                                    test_kind: &TestKind<'tcx>,
                                    test_outcome: usize,
                                    match_pairs: Vec<MatchPair<'tcx>>)
                                    -> BlockAnd<Option<Vec<MatchPair<'tcx>>>> {
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

            let opt_consequent_match_pairs =
                unpack!(block = self.consequent_match_pairs_under_assumption(block,
                                                                             match_pair,
                                                                             test_outcome));
            match opt_consequent_match_pairs {
                None => {
                    // Right kind of test, but wrong outcome. That
                    // means this **entire candidate** is
                    // inapplicable, since the candidate is only
                    // applicable if all of its match-pairs apply (and
                    // this one doesn't).
                    return block.and(None);
                }

                Some(consequent_match_pairs) => {
                    // Test passed; add any new patterns we have to test to the final result.
                    result.extend(consequent_match_pairs)
                }
            }
        }
        block.and(Some(result))
    }

    /// Identifies what test is needed to decide if `match_pair` is applicable.
    ///
    /// It is a bug to call this with a simplifyable pattern.
    pub fn consequent_match_pairs_under_assumption(&mut self,
                                                   mut block: BasicBlock,
                                                   match_pair: MatchPair<'tcx>,
                                                   test_outcome: usize)
                                                   -> BlockAnd<Option<Vec<MatchPair<'tcx>>>> {
        match match_pair.pattern.kind {
            PatternKind::Variant { adt_def, variant_index, subpatterns } => {
                if test_outcome != variant_index {
                    return block.and(None);
                }

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
                block.and(Some(consequent_match_pairs))
            }

            PatternKind::Constant { .. } |
            PatternKind::Range { .. } => {
                // these are boolean tests: if we are on the 0th
                // successor, then they passed, and otherwise they
                // failed, but there are never any more tests to come.
                if test_outcome == 0 {
                    block.and(Some(vec![]))
                } else {
                    block.and(None)
                }
            }

            PatternKind::Slice { prefix, slice, suffix } => {
                if test_outcome == 0 {
                    let mut consequent_match_pairs = vec![];
                    unpack!(block = self.prefix_suffix_slice(&mut consequent_match_pairs,
                                                             block,
                                                             match_pair.lvalue,
                                                             prefix,
                                                             slice,
                                                             suffix));
                    block.and(Some(consequent_match_pairs))
                } else {
                    block.and(None)
                }
            }

            PatternKind::Array { .. } |
            PatternKind::Wild |
            PatternKind::Binding { .. } |
            PatternKind::Leaf { .. } |
            PatternKind::Deref { .. } => {
                self.error_simplifyable(&match_pair)
            }
        }
    }

    fn error_simplifyable(&mut self, match_pair: &MatchPair<'tcx>) -> ! {
        self.hir.span_bug(match_pair.pattern.span,
                          &format!("simplifyable pattern found: {:?}", match_pair.pattern))
    }
}
