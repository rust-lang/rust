// Testing candidates
//
// After candidates have been simplified, the only match pairs that
// remain are those that require some sort of test. The functions here
// identify what tests are needed, perform the tests, and then filter
// the candidates based on the result.

use crate::build::Builder;
use crate::build::matches::{Candidate, MatchPair, Test, TestKind};
use crate::hair::*;
use crate::hair::pattern::compare_const_vals;
use rustc_data_structures::bit_set::BitSet;
use rustc_data_structures::fx::FxHashMap;
use rustc::ty::{self, Ty};
use rustc::ty::util::IntTypeExt;
use rustc::ty::layout::VariantIdx;
use rustc::mir::*;
use rustc::hir::{RangeEnd, Mutability};
use syntax_pos::Span;
use std::cmp::Ordering;

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    /// Identifies what test is needed to decide if `match_pair` is applicable.
    ///
    /// It is a bug to call this with a simplifiable pattern.
    pub fn test<'pat>(&mut self, match_pair: &MatchPair<'pat, 'tcx>) -> Test<'tcx> {
        match *match_pair.pattern.kind {
            PatternKind::Variant { ref adt_def, substs: _, variant_index: _, subpatterns: _ } => {
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Switch {
                        adt_def: adt_def.clone(),
                        variants: BitSet::new_empty(adt_def.variants.len()),
                    },
                }
            }

            PatternKind::Constant { .. } if is_switch_ty(match_pair.pattern.ty) => {
                // For integers, we use a `SwitchInt` match, which allows
                // us to handle more cases.
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::SwitchInt {
                        switch_ty: match_pair.pattern.ty,

                        // these maps are empty to start; cases are
                        // added below in add_cases_to_switch
                        options: vec![],
                        indices: Default::default(),
                    }
                }
            }

            PatternKind::Constant { value } => {
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Eq {
                        value,
                        ty: match_pair.pattern.ty.clone()
                    }
                }
            }

            PatternKind::Range(range) => {
                assert!(range.ty == match_pair.pattern.ty);
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::Range(range),
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

            PatternKind::AscribeUserType { .. } |
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
                                     test_place: &Place<'tcx>,
                                     candidate: &Candidate<'pat, 'tcx>,
                                     switch_ty: Ty<'tcx>,
                                     options: &mut Vec<u128>,
                                     indices: &mut FxHashMap<ty::Const<'tcx>, usize>)
                                     -> bool
    {
        let match_pair = match candidate.match_pairs.iter().find(|mp| mp.place == *test_place) {
            Some(match_pair) => match_pair,
            _ => { return false; }
        };

        match *match_pair.pattern.kind {
            PatternKind::Constant { value } => {
                let switch_ty = ty::ParamEnv::empty().and(switch_ty);
                indices.entry(value)
                       .or_insert_with(|| {
                           options.push(value.unwrap_bits(self.hir.tcx(), switch_ty));
                           options.len() - 1
                       });
                true
            }
            PatternKind::Variant { .. } => {
                panic!("you should have called add_variants_to_switch instead!");
            }
            PatternKind::Range(range) => {
                // Check that none of the switch values are in the range.
                self.values_not_contained_in_range(range, indices)
                    .unwrap_or(false)
            }
            PatternKind::Slice { .. } |
            PatternKind::Array { .. } |
            PatternKind::Wild |
            PatternKind::Binding { .. } |
            PatternKind::AscribeUserType { .. } |
            PatternKind::Leaf { .. } |
            PatternKind::Deref { .. } => {
                // don't know how to add these patterns to a switch
                false
            }
        }
    }

    pub fn add_variants_to_switch<'pat>(&mut self,
                                        test_place: &Place<'tcx>,
                                        candidate: &Candidate<'pat, 'tcx>,
                                        variants: &mut BitSet<VariantIdx>)
                                        -> bool
    {
        let match_pair = match candidate.match_pairs.iter().find(|mp| mp.place == *test_place) {
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
                        place: &Place<'tcx>,
                        test: &Test<'tcx>)
                        -> Vec<BasicBlock> {
        debug!("perform_test({:?}, {:?}: {:?}, {:?})",
               block,
               place,
               place.ty(&self.local_decls, self.hir.tcx()),
               test);
        let source_info = self.source_info(test.span);
        match test.kind {
            TestKind::Switch { adt_def, ref variants } => {
                // Variants is a BitVec of indexes into adt_def.variants.
                let num_enum_variants = adt_def.variants.len();
                let used_variants = variants.count();
                let mut otherwise_block = None;
                let mut target_blocks = Vec::with_capacity(num_enum_variants);
                let mut targets = Vec::with_capacity(used_variants + 1);
                let mut values = Vec::with_capacity(used_variants);
                let tcx = self.hir.tcx();
                for (idx, discr) in adt_def.discriminants(tcx) {
                    target_blocks.push(if variants.contains(idx) {
                        values.push(discr.val);
                        let block = self.cfg.start_new_block();
                        targets.push(block);
                        block
                    } else {
                        *otherwise_block
                            .get_or_insert_with(|| self.cfg.start_new_block())
                    });
                }
                targets.push(
                    otherwise_block
                        .unwrap_or_else(|| self.unreachable_block()),
                );
                debug!("num_enum_variants: {}, tested variants: {:?}, variants: {:?}",
                       num_enum_variants, values, variants);
                let discr_ty = adt_def.repr.discr_type().to_ty(tcx);
                let discr = self.temp(discr_ty, test.span);
                self.cfg.push_assign(block, source_info, &discr,
                                     Rvalue::Discriminant(place.clone()));
                assert_eq!(values.len() + 1, targets.len());
                self.cfg.terminate(block, source_info, TerminatorKind::SwitchInt {
                    discr: Operand::Move(discr),
                    switch_ty: discr_ty,
                    values: From::from(values),
                    targets,
                });
                target_blocks
            }

            TestKind::SwitchInt { switch_ty, ref options, indices: _ } => {
                let (ret, terminator) = if switch_ty.sty == ty::Bool {
                    assert!(options.len() > 0 && options.len() <= 2);
                    let (true_bb, false_bb) = (self.cfg.start_new_block(),
                                               self.cfg.start_new_block());
                    let ret = match options[0] {
                        1 => vec![true_bb, false_bb],
                        0 => vec![false_bb, true_bb],
                        v => span_bug!(test.span, "expected boolean value but got {:?}", v)
                    };
                    (ret, TerminatorKind::if_(self.hir.tcx(), Operand::Copy(place.clone()),
                                              true_bb, false_bb))
                } else {
                    // The switch may be inexhaustive so we
                    // add a catch all block
                    let otherwise = self.cfg.start_new_block();
                    let targets: Vec<_> =
                        options.iter()
                               .map(|_| self.cfg.start_new_block())
                               .chain(Some(otherwise))
                               .collect();
                    (targets.clone(), TerminatorKind::SwitchInt {
                        discr: Operand::Copy(place.clone()),
                        switch_ty,
                        values: options.clone().into(),
                        targets,
                    })
                };
                self.cfg.terminate(block, source_info, terminator);
                ret
            }

            TestKind::Eq { value, mut ty } => {
                let val = Operand::Copy(place.clone());
                let mut expect = self.literal_operand(test.span, ty, value);
                // Use `PartialEq::eq` instead of `BinOp::Eq`
                // (the binop can only handle primitives)
                let fail = self.cfg.start_new_block();
                if !ty.is_scalar() {
                    // If we're using `b"..."` as a pattern, we need to insert an
                    // unsizing coercion, as the byte string has the type `&[u8; N]`.
                    //
                    // We want to do this even when the scrutinee is a reference to an
                    // array, so we can call `<[u8]>::eq` rather than having to find an
                    // `<[u8; N]>::eq`.
                    let unsize = |ty: Ty<'tcx>| match ty.sty {
                        ty::Ref(region, rty, _) => match rty.sty {
                            ty::Array(inner_ty, n) => Some((region, inner_ty, n)),
                            _ => None,
                        },
                        _ => None,
                    };
                    let opt_ref_ty = unsize(ty);
                    let opt_ref_test_ty = unsize(value.ty);
                    let mut place = place.clone();
                    match (opt_ref_ty, opt_ref_test_ty) {
                        // nothing to do, neither is an array
                        (None, None) => {},
                        (Some((region, elem_ty, _)), _) |
                        (None, Some((region, elem_ty, _))) => {
                            let tcx = self.hir.tcx();
                            // make both a slice
                            ty = tcx.mk_imm_ref(region, tcx.mk_slice(elem_ty));
                            if opt_ref_ty.is_some() {
                                place = self.temp(ty, test.span);
                                self.cfg.push_assign(block, source_info, &place,
                                                    Rvalue::Cast(CastKind::Unsize, val, ty));
                            }
                            if opt_ref_test_ty.is_some() {
                                let array = self.literal_operand(
                                    test.span,
                                    value.ty,
                                    value,
                                );

                                let slice = self.temp(ty, test.span);
                                self.cfg.push_assign(block, source_info, &slice,
                                                    Rvalue::Cast(CastKind::Unsize, array, ty));
                                expect = Operand::Move(slice);
                            }
                        },
                    }
                    let eq_def_id = self.hir.tcx().lang_items().eq_trait().unwrap();
                    let (mty, method) = self.hir.trait_method(eq_def_id, "eq", ty, &[ty.into()]);
                    let method = self.hir.tcx().mk_lazy_const(ty::LazyConst::Evaluated(method));

                    let re_erased = self.hir.tcx().types.re_erased;
                    // take the argument by reference
                    let tam = ty::TypeAndMut {
                        ty,
                        mutbl: Mutability::MutImmutable,
                    };
                    let ref_ty = self.hir.tcx().mk_ref(re_erased, tam);

                    // let lhs_ref_place = &lhs;
                    let ref_rvalue = Rvalue::Ref(re_erased, BorrowKind::Shared, place);
                    let lhs_ref_place = self.temp(ref_ty, test.span);
                    self.cfg.push_assign(block, source_info, &lhs_ref_place, ref_rvalue);
                    let val = Operand::Move(lhs_ref_place);

                    // let rhs_place = rhs;
                    let rhs_place = self.temp(ty, test.span);
                    self.cfg.push_assign(block, source_info, &rhs_place, Rvalue::Use(expect));

                    // let rhs_ref_place = &rhs_place;
                    let ref_rvalue = Rvalue::Ref(re_erased, BorrowKind::Shared, rhs_place);
                    let rhs_ref_place = self.temp(ref_ty, test.span);
                    self.cfg.push_assign(block, source_info, &rhs_ref_place, ref_rvalue);
                    let expect = Operand::Move(rhs_ref_place);

                    let bool_ty = self.hir.bool_ty();
                    let eq_result = self.temp(bool_ty, test.span);
                    let eq_block = self.cfg.start_new_block();
                    let cleanup = self.diverge_cleanup();
                    self.cfg.terminate(block, source_info, TerminatorKind::Call {
                        func: Operand::Constant(box Constant {
                            span: test.span,
                            ty: mty,

                            // FIXME(#54571): This constant comes from user
                            // input (a constant in a pattern).  Are
                            // there forms where users can add type
                            // annotations here?  For example, an
                            // associated constant? Need to
                            // experiment.
                            user_ty: None,

                            literal: method,
                        }),
                        args: vec![val, expect],
                        destination: Some((eq_result.clone(), eq_block)),
                        cleanup: Some(cleanup),
                        from_hir_call: false,
                    });

                    // check the result
                    let block = self.cfg.start_new_block();
                    self.cfg.terminate(eq_block, source_info,
                                       TerminatorKind::if_(self.hir.tcx(),
                                                           Operand::Move(eq_result),
                                                           block, fail));
                    vec![block, fail]
                } else {
                    let block = self.compare(block, fail, test.span, BinOp::Eq, expect, val);
                    vec![block, fail]
                }
            }

            TestKind::Range(PatternRange { ref lo, ref hi, ty, ref end }) => {
                // Test `val` by computing `lo <= val && val <= hi`, using primitive comparisons.
                let lo = self.literal_operand(test.span, ty.clone(), lo.clone());
                let hi = self.literal_operand(test.span, ty.clone(), hi.clone());
                let val = Operand::Copy(place.clone());

                let fail = self.cfg.start_new_block();
                let block = self.compare(block, fail, test.span, BinOp::Le, lo, val.clone());
                let block = match *end {
                    RangeEnd::Included => self.compare(block, fail, test.span, BinOp::Le, val, hi),
                    RangeEnd::Excluded => self.compare(block, fail, test.span, BinOp::Lt, val, hi),
                };

                vec![block, fail]
            }

            TestKind::Len { len, op } => {
                let (usize_ty, bool_ty) = (self.hir.usize_ty(), self.hir.bool_ty());
                let (actual, result) = (self.temp(usize_ty, test.span),
                                        self.temp(bool_ty, test.span));

                // actual = len(place)
                self.cfg.push_assign(block, source_info,
                                     &actual, Rvalue::Len(place.clone()));

                // expected = <N>
                let expected = self.push_usize(block, source_info, len);

                // result = actual == expected OR result = actual < expected
                self.cfg.push_assign(block, source_info, &result,
                                     Rvalue::BinaryOp(op,
                                                      Operand::Move(actual),
                                                      Operand::Move(expected)));

                // branch based on result
                let (false_bb, true_bb) = (self.cfg.start_new_block(),
                                           self.cfg.start_new_block());
                self.cfg.terminate(block, source_info,
                                   TerminatorKind::if_(self.hir.tcx(), Operand::Move(result),
                                                       true_bb, false_bb));
                vec![true_bb, false_bb]
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
        let result = self.temp(bool_ty, span);

        // result = op(left, right)
        let source_info = self.source_info(span);
        self.cfg.push_assign(block, source_info, &result,
                             Rvalue::BinaryOp(op, left, right));

        // branch based on result
        let target_block = self.cfg.start_new_block();
        self.cfg.terminate(block, source_info,
                           TerminatorKind::if_(self.hir.tcx(), Operand::Move(result),
                                               target_block, fail_block));
        target_block
    }

    /// Given that we are performing `test` against `test_place`,
    /// this job sorts out what the status of `candidate` will be
    /// after the test. The `resulting_candidates` vector stores, for
    /// each possible outcome of `test`, a vector of the candidates
    /// that will result. This fn should add a (possibly modified)
    /// clone of candidate into `resulting_candidates` wherever
    /// appropriate.
    ///
    /// So, for example, if this candidate is `x @ Some(P0)` and the
    /// Tests is a variant test, then we would add `(x as Option).0 @
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
                                test_place: &Place<'tcx>,
                                test: &Test<'tcx>,
                                candidate: &Candidate<'pat, 'tcx>,
                                resulting_candidates: &mut [Vec<Candidate<'pat, 'tcx>>])
                                -> bool {
        // Find the match_pair for this place (if any). At present,
        // afaik, there can be at most one. (In the future, if we
        // adopted a more general `@` operator, there might be more
        // than one, but it'd be very unusual to have two sides that
        // both require tests; you'd expect one side to be simplified
        // away.)
        let tested_match_pair = candidate.match_pairs.iter()
                                                     .enumerate()
                                                     .find(|&(_, mp)| mp.place == *test_place);
        let (match_pair_index, match_pair) = match tested_match_pair {
            Some(pair) => pair,
            None => {
                // We are not testing this place. Therefore, this
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
                resulting_candidates[variant_index.as_usize()].push(new_candidate);
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

            (&TestKind::SwitchInt { switch_ty: _, ref options, ref indices },
             &PatternKind::Range(range)) => {
                let not_contained = self
                    .values_not_contained_in_range(range, indices)
                    .unwrap_or(false);

                if not_contained {
                    // No switch values are contained in the pattern range,
                    // so the pattern can be matched only if this test fails.
                    let otherwise = options.len();
                    resulting_candidates[otherwise].push(candidate.clone());
                    true
                } else {
                    false
                }
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

            (&TestKind::Range(test),
             &PatternKind::Range(pat)) => {
                if test == pat {
                    resulting_candidates[0]
                        .push(self.candidate_without_match_pair(
                            match_pair_index,
                            candidate,
                        ));
                    return true;
                }

                let no_overlap = (|| {
                    use std::cmp::Ordering::*;
                    use rustc::hir::RangeEnd::*;

                    let param_env = ty::ParamEnv::empty().and(test.ty);
                    let tcx = self.hir.tcx();

                    let lo = compare_const_vals(tcx, test.lo, pat.hi, param_env)?;
                    let hi = compare_const_vals(tcx, test.hi, pat.lo, param_env)?;

                    match (test.end, pat.end, lo, hi) {
                        // pat < test
                        (_, _, Greater, _) |
                        (_, Excluded, Equal, _) |
                        // pat > test
                        (_, _, _, Less) |
                        (Excluded, _, _, Equal) => Some(true),
                        _ => Some(false),
                    }
                })();

                if no_overlap == Some(true) {
                    // Testing range does not overlap with pattern range,
                    // so the pattern can be matched only if this test fails.
                    resulting_candidates[1].push(candidate.clone());
                    true
                } else {
                    false
                }
            }

            (&TestKind::Range(range), &PatternKind::Constant { value }) => {
                if self.const_range_contains(range, value) == Some(false) {
                    // `value` is not contained in the testing range,
                    // so `value` can be matched only if this test fails.
                    resulting_candidates[1].push(candidate.clone());
                    true
                } else {
                    false
                }
            }

            (&TestKind::Range { .. }, _) => false,


            (&TestKind::Eq { .. }, _) |
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
            ascriptions: candidate.ascriptions.clone(),
            guard: candidate.guard.clone(),
            arm_index: candidate.arm_index,
            pat_index: candidate.pat_index,
            pre_binding_block: candidate.pre_binding_block,
            next_candidate_pre_binding_block: candidate.next_candidate_pre_binding_block,
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
            &candidate.match_pairs[match_pair_index].place,
            prefix,
            opt_slice,
            suffix);

        new_candidate
    }

    fn candidate_after_variant_switch<'pat>(&mut self,
                                            match_pair_index: usize,
                                            adt_def: &'tcx ty::AdtDef,
                                            variant_index: VariantIdx,
                                            subpatterns: &'pat [FieldPattern<'tcx>],
                                            candidate: &Candidate<'pat, 'tcx>)
                                            -> Candidate<'pat, 'tcx> {
        let match_pair = &candidate.match_pairs[match_pair_index];

        // So, if we have a match-pattern like `x @ Enum::Variant(P1, P2)`,
        // we want to create a set of derived match-patterns like
        // `(x as Variant).0 @ P1` and `(x as Variant).1 @ P1`.
        let elem = ProjectionElem::Downcast(adt_def, variant_index);
        let downcast_place = match_pair.place.clone().elem(elem); // `(x as Variant)`
        let consequent_match_pairs =
            subpatterns.iter()
                       .map(|subpattern| {
                           // e.g., `(x as Variant).0`
                           let place = downcast_place.clone().field(subpattern.field,
                                                                      subpattern.pattern.ty);
                           // e.g., `(x as Variant).0 @ P1`
                           MatchPair::new(place, &subpattern.pattern)
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
            ascriptions: candidate.ascriptions.clone(),
            guard: candidate.guard.clone(),
            arm_index: candidate.arm_index,
            pat_index: candidate.pat_index,
            pre_binding_block: candidate.pre_binding_block,
            next_candidate_pre_binding_block: candidate.next_candidate_pre_binding_block,
        }
    }

    fn error_simplifyable<'pat>(&mut self, match_pair: &MatchPair<'pat, 'tcx>) -> ! {
        span_bug!(match_pair.pattern.span,
                  "simplifyable pattern found: {:?}",
                  match_pair.pattern)
    }

    fn const_range_contains(
        &self,
        range: PatternRange<'tcx>,
        value: ty::Const<'tcx>,
    ) -> Option<bool> {
        use std::cmp::Ordering::*;

        let param_env = ty::ParamEnv::empty().and(range.ty);
        let tcx = self.hir.tcx();

        let a = compare_const_vals(tcx, range.lo, value, param_env)?;
        let b = compare_const_vals(tcx, value, range.hi, param_env)?;

        match (b, range.end) {
            (Less, _) |
            (Equal, RangeEnd::Included) if a != Greater => Some(true),
            _ => Some(false),
        }
    }

    fn values_not_contained_in_range(
        &self,
        range: PatternRange<'tcx>,
        indices: &FxHashMap<ty::Const<'tcx>, usize>,
    ) -> Option<bool> {
        for &val in indices.keys() {
            if self.const_range_contains(range, val)? {
                return Some(false);
            }
        }

        Some(true)
    }
}

fn is_switch_ty<'tcx>(ty: Ty<'tcx>) -> bool {
    ty.is_integral() || ty.is_char() || ty.is_bool()
}
