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
use rustc_data_structures::fnv::FnvHashMap;
use rustc::middle::const_eval::ConstVal;
use rustc::middle::ty::Ty;
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

            PatternKind::Constant { value: Literal::Value { .. } }
            if is_switch_ty(match_pair.pattern.ty) => {
                // for integers, we use a SwitchInt match, which allows
                // us to handle more cases
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::SwitchInt {
                        switch_ty: match_pair.pattern.ty,

                        // these maps are empty to start; cases are
                        // added below in add_cases_to_switch
                        options: vec![],
                        indices: FnvHashMap(),
                    }
                }
            }

            PatternKind::Constant { ref value } => {
                // for other types, we use an equality comparison
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Eq {
                        value: value.clone(),
                        ty: match_pair.pattern.ty.clone(),
                    }
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

    pub fn add_cases_to_switch(&mut self,
                               test_lvalue: &Lvalue<'tcx>,
                               candidate: &Candidate<'tcx>,
                               switch_ty: Ty<'tcx>,
                               options: &mut Vec<ConstVal>,
                               indices: &mut FnvHashMap<ConstVal, usize>)
    {
        let match_pair = match candidate.match_pairs.iter().find(|mp| mp.lvalue == *test_lvalue) {
            Some(match_pair) => match_pair,
            _ => { return; }
        };

        match match_pair.pattern.kind {
            PatternKind::Constant { value: Literal::Value { ref value } } => {
                // if the lvalues match, the type should match
                assert_eq!(match_pair.pattern.ty, switch_ty);

                indices.entry(value.clone())
                       .or_insert_with(|| {
                           options.push(value.clone());
                           options.len() - 1
                       });
            }

            PatternKind::Range { .. } => {
            }

            PatternKind::Constant { .. } |
            PatternKind::Variant { .. } |
            PatternKind::Slice { .. } |
            PatternKind::Array { .. } |
            PatternKind::Wild |
            PatternKind::Binding { .. } |
            PatternKind::Leaf { .. } |
            PatternKind::Deref { .. } => {
            }
        }
    }

    /// Generates the code to perform a test.
    pub fn perform_test(&mut self,
                        block: BasicBlock,
                        lvalue: &Lvalue<'tcx>,
                        test: &Test<'tcx>)
                        -> Vec<BasicBlock> {
        match test.kind {
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

            TestKind::SwitchInt { switch_ty, ref options, indices: _ } => {
                let otherwise = self.cfg.start_new_block();
                let targets: Vec<_> =
                    options.iter()
                           .map(|_| self.cfg.start_new_block())
                           .chain(Some(otherwise))
                           .collect();
                self.cfg.terminate(block, Terminator::SwitchInt {
                    discr: lvalue.clone(),
                    switch_ty: switch_ty,
                    values: options.clone(),
                    targets: targets.clone(),
                });
                targets
            }

            TestKind::Eq { ref value, ty } => {
                // call PartialEq::eq(discrim, constant)
                let constant = self.literal_operand(test.span, ty.clone(), value.clone());
                let item_ref = self.hir.partial_eq(ty);
                self.call_comparison_fn(block, test.span, item_ref,
                                        Operand::Consume(lvalue.clone()), constant)
            }

            TestKind::Range { ref lo, ref hi, ty } => {
                // Test `v` by computing `PartialOrd::le(lo, v) && PartialOrd::le(v, hi)`.
                let lo = self.literal_operand(test.span, ty.clone(), lo.clone());
                let hi = self.literal_operand(test.span, ty.clone(), hi.clone());
                let item_ref = self.hir.partial_le(ty);

                let lo_blocks = self.call_comparison_fn(block,
                                                        test.span,
                                                        item_ref.clone(),
                                                        lo,
                                                        Operand::Consume(lvalue.clone()));

                let hi_blocks = self.call_comparison_fn(lo_blocks[0],
                                                        test.span,
                                                        item_ref,
                                                        Operand::Consume(lvalue.clone()),
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
                          lvalue1: Operand<'tcx>,
                          lvalue2: Operand<'tcx>)
                          -> Vec<BasicBlock> {
        let target_blocks = vec![self.cfg.start_new_block(), self.cfg.start_new_block()];

        let bool_ty = self.hir.bool_ty();
        let eq_result = self.temp(bool_ty);
        let func = self.item_ref_operand(span, item_ref);
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
                                      test_lvalue: &Lvalue<'tcx>,
                                      test_kind: &TestKind<'tcx>,
                                      test_outcome: usize,
                                      candidate: &Candidate<'tcx>)
                                      -> Option<Candidate<'tcx>> {
        let candidate = candidate.clone();
        let match_pairs = candidate.match_pairs;
        let result = self.match_pairs_under_assumption(test_lvalue,
                                                       test_kind,
                                                       test_outcome,
                                                       match_pairs);
        match result {
            Some(match_pairs) => Some(Candidate { match_pairs: match_pairs, ..candidate }),
            None => None,
        }
    }

    /// Helper for candidate_under_assumption that does the actual
    /// work of transforming the list of match pairs.
    fn match_pairs_under_assumption(&mut self,
                                    test_lvalue: &Lvalue<'tcx>,
                                    test_kind: &TestKind<'tcx>,
                                    test_outcome: usize,
                                    match_pairs: Vec<MatchPair<'tcx>>)
                                    -> Option<Vec<MatchPair<'tcx>>> {
        let mut result = vec![];

        for match_pair in match_pairs {
            // if the match pair is testing a different lvalue, it
            // is unaffected by this test.
            if match_pair.lvalue != *test_lvalue {
                result.push(match_pair);
                continue;
            }

            // if this test doesn't tell us anything about this match-pair, then hang onto it.
            if !self.test_informs_match_pair(&match_pair, test_kind, test_outcome) {
                result.push(match_pair);
                continue;
            }

            // otherwise, build up the consequence match pairs
            let opt_consequent_match_pairs =
                self.consequent_match_pairs_under_assumption(match_pair,
                                                             test_kind,
                                                             test_outcome);
            match opt_consequent_match_pairs {
                None => {
                    // Right kind of test, but wrong outcome. That
                    // means this **entire candidate** is
                    // inapplicable, since the candidate is only
                    // applicable if all of its match-pairs apply (and
                    // this one doesn't).
                    return None;
                }

                Some(consequent_match_pairs) => {
                    // Test passed; add any new patterns we have to test to the final result.
                    result.extend(consequent_match_pairs)
                }
            }
        }

        Some(result)
    }

    /// Given that we executed `test` to `match_pair.lvalue` with
    /// outcome `test_outcome`, does that tell us anything about
    /// whether `match_pair` applies?
    ///
    /// Often it does not. For example, if we are testing whether
    /// the discriminant equals 4, and we find that it does not,
    /// but the `match_pair` is testing if the discriminant equals 5,
    /// that does not help us.
    fn test_informs_match_pair(&mut self,
                               match_pair: &MatchPair<'tcx>,
                               test_kind: &TestKind<'tcx>,
                               _test_outcome: usize)
                               -> bool {
        match match_pair.pattern.kind {
            PatternKind::Variant { .. } => {
                match *test_kind {
                    TestKind::Switch { .. } => true,
                    _ => false,
                }
            }

            PatternKind::Constant { value: Literal::Value { .. } }
            if is_switch_ty(match_pair.pattern.ty) => {
                match *test_kind {
                    TestKind::SwitchInt { .. } => true,

                    // Did not do an integer equality test (which is always a SwitchInt).
                    // So we learned nothing relevant to this match-pair.
                    //
                    // TODO we could use TestKind::Range to rule
                    // things out here, in some cases.
                    _ => false,
                }
            }

            PatternKind::Constant { .. } |
            PatternKind::Range { .. } |
            PatternKind::Slice { .. } => {
                let pattern_test = self.test(&match_pair);
                if pattern_test.kind == *test_kind {
                    true
                } else {
                    // TODO in all 3 cases, we could sometimes do
                    // better here. For example, if we are checking
                    // whether the value is equal to X, and we find
                    // that it is, that (may) imply value is not equal
                    // to Y. Or, if the range tested is `3..5`, and
                    // our range is `4..5`, then we know that our
                    // range also does not apply. Similarly, if we
                    // test that length is >= 5, and it fails, we also
                    // know that length is not >= 7. etc.
                    false
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

    /// Given that we executed `test` with outcome `test_outcome`,
    /// what are the resulting match pairs? This can return either:
    ///
    /// - None, meaning that the test indicated that this outcome
    ///   means that this match-pair is not the current one for the
    ///   current discriminant (which rules out the enclosing
    ///   candidate);
    /// - Some(...), meaning that either the test didn't tell us whether this
    ///   match-pair is correct or not, or that we DID match and now have
    ///   subsequent matches to perform.
    ///
    /// As an example, consider:
    ///
    /// ```
    /// match option {
    ///     Ok(<pattern>) => ...,
    ///     Err(_) => ...,
    /// }
    /// ```
    ///
    /// Suppose that the `test` is a `Switch` and the outcome is
    /// `Ok`. Then in that case, the first arm will have a match-pair
    /// of `option @ Ok(<pattern>)`. In that case we will return
    /// `Some(vec![(option as Ok) @ <pattern>])`. The `Some` reuslt
    /// indicates that the match-pair still applies, and we now have
    /// to test `(option as Ok) @ <pattern>`.
    ///
    /// On the second arm, a `None` will be returned, because if we
    /// observed that `option` has the discriminant `Ok`, then the
    /// second arm cannot apply.
    pub fn consequent_match_pairs_under_assumption(&mut self,
                                                   match_pair: MatchPair<'tcx>,
                                                   test_kind: &TestKind<'tcx>,
                                                   test_outcome: usize)
                                                   -> Option<Vec<MatchPair<'tcx>>> {
        match match_pair.pattern.kind {
            PatternKind::Variant { adt_def, variant_index, subpatterns } => {
                assert!(match *test_kind { TestKind::Switch { .. } => true,
                                           _ => false });

                if test_outcome != variant_index {
                    return None; // Tested, but found the wrong variant.
                }

                // Correct variant. Extract the subitems and match
                // those. The lvalue goes gets downcast, so
                // e.g. `foo.bar` becomes `foo.bar as Variant`.
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
                Some(consequent_match_pairs)
            }

            PatternKind::Constant { value: Literal::Value { ref value } }
            if is_switch_ty(match_pair.pattern.ty) => {
                match *test_kind {
                    TestKind::SwitchInt { switch_ty: _, options: _, ref indices } => {
                        let index = indices[value];
                        if index == test_outcome {
                            Some(vec![]) // this value, nothing left to test
                        } else {
                            None // some other value, candidate is inapplicable
                        }
                    }

                    _ => {
                        self.hir.span_bug(
                            match_pair.pattern.span,
                            &format!("did a switch-int, but value {:?} not found in cases",
                                     value));
                    }
                }
            }

            PatternKind::Constant { .. } |
            PatternKind::Range { .. } |
            PatternKind::Slice { .. } => {
                if test_outcome == 0 {
                    Some(vec![])
                } else {
                    None
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

fn is_switch_ty<'tcx>(ty: Ty<'tcx>) -> bool {
    ty.is_integral() || ty.is_char()
}
