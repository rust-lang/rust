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
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::bitvec::BitVector;
use rustc::middle::const_val::ConstVal;
use rustc::ty::{self, Ty};
use rustc::mir::*;
use syntax_pos::Span;
use std::cmp::Ordering;

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    /// Identifies what test is needed to decide if `match_pair` is applicable.
    ///
    /// It is a bug to call this with a simplifyable pattern.
    pub fn test<'pat>(&mut self, match_pair: &MatchPair<'pat, 'tcx>) -> Test<'tcx> {
        match *match_pair.pattern.kind {
            PatternKind::Variant { ref adt_def, substs: _, variant_index: _, subpatterns: _ } => {
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Switch {
                        adt_def: adt_def.clone(),
                        variants: BitVector::new(self.hir.num_variants(adt_def)),
                    },
                }
            }

            PatternKind::Constant { .. }
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
                        indices: FxHashMap(),
                    }
                }
            }

            PatternKind::Constant { ref value } => {
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Eq {
                        value: value.clone(),
                        ty: match_pair.pattern.ty.clone()
                    }
                }
            }

            PatternKind::Range { ref lo, ref hi } => {
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Range {
                        lo: Literal::Value { value: lo.clone() },
                        hi: Literal::Value { value: hi.clone() },
                        ty: match_pair.pattern.ty.clone(),
                    },
                }
            }

            PatternKind::Slice { ref prefix, ref slice, ref suffix }
                    if !match_pair.slice_len_checked => {
                let len = prefix.len() + suffix.len();
                let op = if slice.is_some() {
                    BinOp::Ge
                } else {
                    BinOp::Eq
                };
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Len { len: len as u64, op: op },
                }
            }

            PatternKind::Array { .. } |
            PatternKind::Slice { .. } |
            PatternKind::Wild |
            PatternKind::Binding { .. } |
            PatternKind::Leaf { .. } |
            PatternKind::Deref { .. } => {
                self.error_simplifyable(match_pair)
            }
        }
    }

    pub fn add_cases_to_switch<'pat>(&mut self,
                                     test_lvalue: &Lvalue<'tcx>,
                                     candidate: &Candidate<'pat, 'tcx>,
                                     switch_ty: Ty<'tcx>,
                                     options: &mut Vec<ConstVal>,
                                     indices: &mut FxHashMap<ConstVal, usize>)
                                     -> bool
    {
        let match_pair = match candidate.match_pairs.iter().find(|mp| mp.lvalue == *test_lvalue) {
            Some(match_pair) => match_pair,
            _ => { return false; }
        };

        match *match_pair.pattern.kind {
            PatternKind::Constant { ref value } => {
                // if the lvalues match, the type should match
                assert_eq!(match_pair.pattern.ty, switch_ty);

                indices.entry(value.clone())
                       .or_insert_with(|| {
                           options.push(value.clone());
                           options.len() - 1
                       });
                true
            }
            PatternKind::Variant { .. } => {
                panic!("you should have called add_variants_to_switch instead!");
            }
            PatternKind::Range { .. } |
            PatternKind::Slice { .. } |
            PatternKind::Array { .. } |
            PatternKind::Wild |
            PatternKind::Binding { .. } |
            PatternKind::Leaf { .. } |
            PatternKind::Deref { .. } => {
                // don't know how to add these patterns to a switch
                false
            }
        }
    }

    pub fn add_variants_to_switch<'pat>(&mut self,
                                        test_lvalue: &Lvalue<'tcx>,
                                        candidate: &Candidate<'pat, 'tcx>,
                                        variants: &mut BitVector)
                                        -> bool
    {
        let match_pair = match candidate.match_pairs.iter().find(|mp| mp.lvalue == *test_lvalue) {
            Some(match_pair) => match_pair,
            _ => { return false; }
        };

        match *match_pair.pattern.kind {
            PatternKind::Variant { adt_def: _ , variant_index,  .. } => {
                // We have a pattern testing for variant `variant_index`
                // set the corresponding index to true
                variants.insert(variant_index);
                true
            }
            _ => {
                // don't know how to add these patterns to a switch
                false
            }
        }
    }

    /// Generates the code to perform a test.
    pub fn perform_test(&mut self,
                        block: BasicBlock,
                        lvalue: &Lvalue<'tcx>,
                        test: &Test<'tcx>)
                        -> Vec<BasicBlock> {
        let source_info = self.source_info(test.span);
        match test.kind {
            TestKind::Switch { adt_def, ref variants } => {
                let num_enum_variants = self.hir.num_variants(adt_def);
                let mut otherwise_block = None;
                let target_blocks: Vec<_> = (0..num_enum_variants).map(|i| {
                    if variants.contains(i) {
                        self.cfg.start_new_block()
                    } else {
                        if otherwise_block.is_none() {
                            otherwise_block = Some(self.cfg.start_new_block());
                        }
                        otherwise_block.unwrap()
                    }
                }).collect();
                debug!("num_enum_variants: {}, num tested variants: {}, variants: {:?}",
                       num_enum_variants, variants.iter().count(), variants);
                self.cfg.terminate(block, source_info, TerminatorKind::Switch {
                    discr: lvalue.clone(),
                    adt_def: adt_def,
                    targets: target_blocks.clone()
                });
                target_blocks
            }

            TestKind::SwitchInt { switch_ty, ref options, indices: _ } => {
                let (targets, term) = match switch_ty.sty {
                    // If we're matching on boolean we can
                    // use the If TerminatorKind instead
                    ty::TyBool => {
                        assert!(options.len() > 0 && options.len() <= 2);

                        let (true_bb, else_bb) =
                            (self.cfg.start_new_block(),
                             self.cfg.start_new_block());

                        let targets = match &options[0] {
                            &ConstVal::Bool(true) => vec![true_bb, else_bb],
                            &ConstVal::Bool(false) => vec![else_bb, true_bb],
                            v => span_bug!(test.span, "expected boolean value but got {:?}", v)
                        };

                        (targets,
                         TerminatorKind::If {
                             cond: Operand::Consume(lvalue.clone()),
                             targets: (true_bb, else_bb)
                         })

                    }
                    _ => {
                        // The switch may be inexhaustive so we
                        // add a catch all block
                        let otherwise = self.cfg.start_new_block();
                        let targets: Vec<_> =
                            options.iter()
                                   .map(|_| self.cfg.start_new_block())
                                   .chain(Some(otherwise))
                                   .collect();

                        (targets.clone(),
                         TerminatorKind::SwitchInt {
                             discr: lvalue.clone(),
                             switch_ty: switch_ty,
                             values: options.clone(),
                             targets: targets
                         })
                    }
                };

                self.cfg.terminate(block, source_info, term);
                targets
            }

            TestKind::Eq { ref value, mut ty } => {
                let mut val = Operand::Consume(lvalue.clone());

                // If we're using b"..." as a pattern, we need to insert an
                // unsizing coercion, as the byte string has the type &[u8; N].
                let expect = if let ConstVal::ByteStr(ref bytes) = *value {
                    let tcx = self.hir.tcx();

                    // Unsize the lvalue to &[u8], too, if necessary.
                    if let ty::TyRef(region, mt) = ty.sty {
                        if let ty::TyArray(_, _) = mt.ty.sty {
                            ty = tcx.mk_imm_ref(region, tcx.mk_slice(tcx.types.u8));
                            let val_slice = self.temp(ty);
                            self.cfg.push_assign(block, source_info, &val_slice,
                                                 Rvalue::Cast(CastKind::Unsize, val, ty));
                            val = Operand::Consume(val_slice);
                        }
                    }

                    assert!(ty.is_slice());

                    let array_ty = tcx.mk_array(tcx.types.u8, bytes.len());
                    let array_ref = tcx.mk_imm_ref(tcx.mk_region(ty::ReStatic), array_ty);
                    let array = self.literal_operand(test.span, array_ref, Literal::Value {
                        value: value.clone()
                    });

                    let slice = self.temp(ty);
                    self.cfg.push_assign(block, source_info, &slice,
                                         Rvalue::Cast(CastKind::Unsize, array, ty));
                    Operand::Consume(slice)
                } else {
                    self.literal_operand(test.span, ty, Literal::Value {
                        value: value.clone()
                    })
                };

                // Use PartialEq::eq for &str and &[u8] slices, instead of BinOp::Eq.
                let fail = self.cfg.start_new_block();
                if let ty::TyRef(_, mt) = ty.sty {
                    assert!(ty.is_slice());
                    let eq_def_id = self.hir.tcx().lang_items.eq_trait().unwrap();
                    let ty = mt.ty;
                    let (mty, method) = self.hir.trait_method(eq_def_id, "eq", ty, &[ty]);

                    let bool_ty = self.hir.bool_ty();
                    let eq_result = self.temp(bool_ty);
                    let eq_block = self.cfg.start_new_block();
                    let cleanup = self.diverge_cleanup();
                    self.cfg.terminate(block, source_info, TerminatorKind::Call {
                        func: Operand::Constant(Constant {
                            span: test.span,
                            ty: mty,
                            literal: method
                        }),
                        args: vec![val, expect],
                        destination: Some((eq_result.clone(), eq_block)),
                        cleanup: cleanup,
                    });

                    // check the result
                    let block = self.cfg.start_new_block();
                    self.cfg.terminate(eq_block, source_info, TerminatorKind::If {
                        cond: Operand::Consume(eq_result),
                        targets: (block, fail),
                    });

                    vec![block, fail]
                } else {
                    let block = self.compare(block, fail, test.span, BinOp::Eq, expect, val);
                    vec![block, fail]
                }
            }

            TestKind::Range { ref lo, ref hi, ty } => {
                // Test `val` by computing `lo <= val && val <= hi`, using primitive comparisons.
                let lo = self.literal_operand(test.span, ty.clone(), lo.clone());
                let hi = self.literal_operand(test.span, ty.clone(), hi.clone());
                let val = Operand::Consume(lvalue.clone());

                let fail = self.cfg.start_new_block();
                let block = self.compare(block, fail, test.span, BinOp::Le, lo, val.clone());
                let block = self.compare(block, fail, test.span, BinOp::Le, val, hi);

                vec![block, fail]
            }

            TestKind::Len { len, op } => {
                let (usize_ty, bool_ty) = (self.hir.usize_ty(), self.hir.bool_ty());
                let (actual, result) = (self.temp(usize_ty), self.temp(bool_ty));

                // actual = len(lvalue)
                self.cfg.push_assign(block, source_info,
                                     &actual, Rvalue::Len(lvalue.clone()));

                // expected = <N>
                let expected = self.push_usize(block, source_info, len);

                // result = actual == expected OR result = actual < expected
                self.cfg.push_assign(block, source_info, &result,
                                     Rvalue::BinaryOp(op,
                                                      Operand::Consume(actual),
                                                      Operand::Consume(expected)));

                // branch based on result
                let target_blocks: Vec<_> = vec![self.cfg.start_new_block(),
                                                 self.cfg.start_new_block()];
                self.cfg.terminate(block, source_info, TerminatorKind::If {
                    cond: Operand::Consume(result),
                    targets: (target_blocks[0], target_blocks[1])
                });

                target_blocks
            }
        }
    }

    fn compare(&mut self,
               block: BasicBlock,
               fail_block: BasicBlock,
               span: Span,
               op: BinOp,
               left: Operand<'tcx>,
               right: Operand<'tcx>) -> BasicBlock {
        let bool_ty = self.hir.bool_ty();
        let result = self.temp(bool_ty);

        // result = op(left, right)
        let source_info = self.source_info(span);
        self.cfg.push_assign(block, source_info, &result,
                             Rvalue::BinaryOp(op, left, right));

        // branch based on result
        let target_block = self.cfg.start_new_block();
        self.cfg.terminate(block, source_info, TerminatorKind::If {
            cond: Operand::Consume(result),
            targets: (target_block, fail_block)
        });

        target_block
    }

    /// Given that we are performing `test` against `test_lvalue`,
    /// this job sorts out what the status of `candidate` will be
    /// after the test. The `resulting_candidates` vector stores, for
    /// each possible outcome of `test`, a vector of the candidates
    /// that will result. This fn should add a (possibly modified)
    /// clone of candidate into `resulting_candidates` wherever
    /// appropriate.
    ///
    /// So, for example, if this candidate is `x @ Some(P0)` and the
    /// test is a variant test, then we would add `(x as Option).0 @
    /// P0` to the `resulting_candidates` entry corresponding to the
    /// variant `Some`.
    ///
    /// However, in some cases, the test may just not be relevant to
    /// candidate. For example, suppose we are testing whether `foo.x == 22`,
    /// but in one match arm we have `Foo { x: _, ... }`... in that case,
    /// the test for what value `x` has has no particular relevance
    /// to this candidate. In such cases, this function just returns false
    /// without doing anything. This is used by the overall `match_candidates`
    /// algorithm to structure the match as a whole. See `match_candidates` for
    /// more details.
    ///
    /// FIXME(#29623). In some cases, we have some tricky choices to
    /// make.  for example, if we are testing that `x == 22`, but the
    /// candidate is `x @ 13..55`, what should we do? In the event
    /// that the test is true, we know that the candidate applies, but
    /// in the event of false, we don't know that it *doesn't*
    /// apply. For now, we return false, indicate that the test does
    /// not apply to this candidate, but it might be we can get
    /// tighter match code if we do something a bit different.
    pub fn sort_candidate<'pat>(&mut self,
                                test_lvalue: &Lvalue<'tcx>,
                                test: &Test<'tcx>,
                                candidate: &Candidate<'pat, 'tcx>,
                                resulting_candidates: &mut [Vec<Candidate<'pat, 'tcx>>])
                                -> bool {
        // Find the match_pair for this lvalue (if any). At present,
        // afaik, there can be at most one. (In the future, if we
        // adopted a more general `@` operator, there might be more
        // than one, but it'd be very unusual to have two sides that
        // both require tests; you'd expect one side to be simplified
        // away.)
        let tested_match_pair = candidate.match_pairs.iter()
                                                     .enumerate()
                                                     .filter(|&(_, mp)| mp.lvalue == *test_lvalue)
                                                     .next();
        let (match_pair_index, match_pair) = match tested_match_pair {
            Some(pair) => pair,
            None => {
                // We are not testing this lvalue. Therefore, this
                // candidate applies to ALL outcomes.
                return false;
            }
        };

        match (&test.kind, &*match_pair.pattern.kind) {
            // If we are performing a variant switch, then this
            // informs variant patterns, but nothing else.
            (&TestKind::Switch { adt_def: tested_adt_def, .. },
             &PatternKind::Variant { adt_def, variant_index, ref subpatterns, .. }) => {
                assert_eq!(adt_def, tested_adt_def);
                let new_candidate =
                    self.candidate_after_variant_switch(match_pair_index,
                                                        adt_def,
                                                        variant_index,
                                                        subpatterns,
                                                        candidate);
                resulting_candidates[variant_index].push(new_candidate);
                true
            }
            (&TestKind::Switch { .. }, _) => false,

            // If we are performing a switch over integers, then this informs integer
            // equality, but nothing else.
            //
            // FIXME(#29623) we could use PatternKind::Range to rule
            // things out here, in some cases.
            (&TestKind::SwitchInt { switch_ty: _, options: _, ref indices },
             &PatternKind::Constant { ref value })
            if is_switch_ty(match_pair.pattern.ty) => {
                let index = indices[value];
                let new_candidate = self.candidate_without_match_pair(match_pair_index,
                                                                      candidate);
                resulting_candidates[index].push(new_candidate);
                true
            }
            (&TestKind::SwitchInt { .. }, _) => false,


            (&TestKind::Len { len: test_len, op: BinOp::Eq },
             &PatternKind::Slice { ref prefix, ref slice, ref suffix }) => {
                let pat_len = (prefix.len() + suffix.len()) as u64;
                match (test_len.cmp(&pat_len), slice) {
                    (Ordering::Equal, &None) => {
                        // on true, min_len = len = $actual_length,
                        // on false, len != $actual_length
                        resulting_candidates[0].push(
                            self.candidate_after_slice_test(match_pair_index,
                                                            candidate,
                                                            prefix,
                                                            slice.as_ref(),
                                                            suffix)
                        );
                        true
                    }
                    (Ordering::Less, _) => {
                        // test_len < pat_len. If $actual_len = test_len,
                        // then $actual_len < pat_len and we don't have
                        // enough elements.
                        resulting_candidates[1].push(candidate.clone());
                        true
                    }
                    (Ordering::Equal, &Some(_)) | (Ordering::Greater, &Some(_)) => {
                        // This can match both if $actual_len = test_len >= pat_len,
                        // and if $actual_len > test_len. We can't advance.
                        false
                    }
                    (Ordering::Greater, &None) => {
                        // test_len != pat_len, so if $actual_len = test_len, then
                        // $actual_len != pat_len.
                        resulting_candidates[1].push(candidate.clone());
                        true
                    }
                }
            }

            (&TestKind::Len { len: test_len, op: BinOp::Ge },
             &PatternKind::Slice { ref prefix, ref slice, ref suffix }) => {
                // the test is `$actual_len >= test_len`
                let pat_len = (prefix.len() + suffix.len()) as u64;
                match (test_len.cmp(&pat_len), slice) {
                    (Ordering::Equal, &Some(_))  => {
                        // $actual_len >= test_len = pat_len,
                        // so we can match.
                        resulting_candidates[0].push(
                            self.candidate_after_slice_test(match_pair_index,
                                                            candidate,
                                                            prefix,
                                                            slice.as_ref(),
                                                            suffix)
                        );
                        true
                    }
                    (Ordering::Less, _) | (Ordering::Equal, &None) => {
                        // test_len <= pat_len. If $actual_len < test_len,
                        // then it is also < pat_len, so the test passing is
                        // necessary (but insufficient).
                        resulting_candidates[0].push(candidate.clone());
                        true
                    }
                    (Ordering::Greater, &None) => {
                        // test_len > pat_len. If $actual_len >= test_len > pat_len,
                        // then we know we won't have a match.
                        resulting_candidates[1].push(candidate.clone());
                        true
                    }
                    (Ordering::Greater, &Some(_)) => {
                        // test_len < pat_len, and is therefore less
                        // strict. This can still go both ways.
                        false
                    }
                }
            }

            (&TestKind::Eq { .. }, _) |
            (&TestKind::Range { .. }, _) |
            (&TestKind::Len { .. }, _) => {
                // These are all binary tests.
                //
                // FIXME(#29623) we can be more clever here
                let pattern_test = self.test(&match_pair);
                if pattern_test.kind == test.kind {
                    let new_candidate = self.candidate_without_match_pair(match_pair_index,
                                                                          candidate);
                    resulting_candidates[0].push(new_candidate);
                    true
                } else {
                    false
                }
            }
        }
    }

    fn candidate_without_match_pair<'pat>(&mut self,
                                          match_pair_index: usize,
                                          candidate: &Candidate<'pat, 'tcx>)
                                          -> Candidate<'pat, 'tcx> {
        let other_match_pairs =
            candidate.match_pairs.iter()
                                 .enumerate()
                                 .filter(|&(index, _)| index != match_pair_index)
                                 .map(|(_, mp)| mp.clone())
                                 .collect();
        Candidate {
            span: candidate.span,
            match_pairs: other_match_pairs,
            bindings: candidate.bindings.clone(),
            guard: candidate.guard.clone(),
            arm_index: candidate.arm_index,
        }
    }

    fn candidate_after_slice_test<'pat>(&mut self,
                                        match_pair_index: usize,
                                        candidate: &Candidate<'pat, 'tcx>,
                                        prefix: &'pat [Pattern<'tcx>],
                                        opt_slice: Option<&'pat Pattern<'tcx>>,
                                        suffix: &'pat [Pattern<'tcx>])
                                        -> Candidate<'pat, 'tcx> {
        let mut new_candidate =
            self.candidate_without_match_pair(match_pair_index, candidate);
        self.prefix_slice_suffix(
            &mut new_candidate.match_pairs,
            &candidate.match_pairs[match_pair_index].lvalue,
            prefix,
            opt_slice,
            suffix);

        new_candidate
    }

    fn candidate_after_variant_switch<'pat>(&mut self,
                                            match_pair_index: usize,
                                            adt_def: &'tcx ty::AdtDef,
                                            variant_index: usize,
                                            subpatterns: &'pat [FieldPattern<'tcx>],
                                            candidate: &Candidate<'pat, 'tcx>)
                                            -> Candidate<'pat, 'tcx> {
        let match_pair = &candidate.match_pairs[match_pair_index];

        // So, if we have a match-pattern like `x @ Enum::Variant(P1, P2)`,
        // we want to create a set of derived match-patterns like
        // `(x as Variant).0 @ P1` and `(x as Variant).1 @ P1`.
        let elem = ProjectionElem::Downcast(adt_def, variant_index);
        let downcast_lvalue = match_pair.lvalue.clone().elem(elem); // `(x as Variant)`
        let consequent_match_pairs =
            subpatterns.iter()
                       .map(|subpattern| {
                           // e.g., `(x as Variant).0`
                           let lvalue = downcast_lvalue.clone().field(subpattern.field,
                                                                      subpattern.pattern.ty);
                           // e.g., `(x as Variant).0 @ P1`
                           MatchPair::new(lvalue, &subpattern.pattern)
                       });

        // In addition, we need all the other match pairs from the old candidate.
        let other_match_pairs =
            candidate.match_pairs.iter()
                                 .enumerate()
                                 .filter(|&(index, _)| index != match_pair_index)
                                 .map(|(_, mp)| mp.clone());

        let all_match_pairs = consequent_match_pairs.chain(other_match_pairs).collect();

        Candidate {
            span: candidate.span,
            match_pairs: all_match_pairs,
            bindings: candidate.bindings.clone(),
            guard: candidate.guard.clone(),
            arm_index: candidate.arm_index,
        }
    }

    fn error_simplifyable<'pat>(&mut self, match_pair: &MatchPair<'pat, 'tcx>) -> ! {
        span_bug!(match_pair.pattern.span,
                  "simplifyable pattern found: {:?}",
                  match_pair.pattern)
    }
}

fn is_switch_ty<'tcx>(ty: Ty<'tcx>) -> bool {
    ty.is_integral() || ty.is_char() || ty.is_bool()
}
