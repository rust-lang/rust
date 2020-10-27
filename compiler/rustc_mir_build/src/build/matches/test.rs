// Testing candidates
//
// After candidates have been simplified, the only match pairs that
// remain are those that require some sort of test. The functions here
// identify what tests are needed, perform the tests, and then filter
// the candidates based on the result.

use crate::build::matches::{Candidate, MatchPair, Test, TestKind};
use crate::build::Builder;
use crate::thir::pattern::compare_const_vals;
use crate::thir::*;
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::{LangItem, RangeEnd};
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::*;
use rustc_middle::ty::util::IntTypeExt;
use rustc_middle::ty::{self, adjustment::PointerCast, Ty};
use rustc_span::symbol::sym;
use rustc_target::abi::VariantIdx;

use std::cmp::Ordering;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Identifies what test is needed to decide if `match_pair` is applicable.
    ///
    /// It is a bug to call this with a simplifiable pattern.
    pub(super) fn test<'pat>(&mut self, match_pair: &MatchPair<'pat, 'tcx>) -> Test<'tcx> {
        match *match_pair.pattern.kind {
            PatKind::Variant { ref adt_def, substs: _, variant_index: _, subpatterns: _ } => Test {
                span: match_pair.pattern.span,
                kind: TestKind::Switch {
                    adt_def,
                    variants: BitSet::new_empty(adt_def.variants.len()),
                },
            },

            PatKind::Constant { .. } if is_switch_ty(match_pair.pattern.ty) => {
                // For integers, we use a `SwitchInt` match, which allows
                // us to handle more cases.
                Test {
                    span: match_pair.pattern.span,
                    kind: TestKind::SwitchInt {
                        switch_ty: match_pair.pattern.ty,

                        // these maps are empty to start; cases are
                        // added below in add_cases_to_switch
                        options: Default::default(),
                    },
                }
            }

            PatKind::Constant { value } => Test {
                span: match_pair.pattern.span,
                kind: TestKind::Eq { value, ty: match_pair.pattern.ty.clone() },
            },

            PatKind::Range(range) => {
                assert_eq!(range.lo.ty, match_pair.pattern.ty);
                assert_eq!(range.hi.ty, match_pair.pattern.ty);
                Test { span: match_pair.pattern.span, kind: TestKind::Range(range) }
            }

            PatKind::Slice { ref prefix, ref slice, ref suffix } => {
                let len = prefix.len() + suffix.len();
                let op = if slice.is_some() { BinOp::Ge } else { BinOp::Eq };
                Test { span: match_pair.pattern.span, kind: TestKind::Len { len: len as u64, op } }
            }

            PatKind::Or { .. } => bug!("or-patterns should have already been handled"),

            PatKind::AscribeUserType { .. }
            | PatKind::Array { .. }
            | PatKind::Wild
            | PatKind::Binding { .. }
            | PatKind::Leaf { .. }
            | PatKind::Deref { .. } => self.error_simplifyable(match_pair),
        }
    }

    pub(super) fn add_cases_to_switch<'pat>(
        &mut self,
        test_place: &Place<'tcx>,
        candidate: &Candidate<'pat, 'tcx>,
        switch_ty: Ty<'tcx>,
        options: &mut FxIndexMap<&'tcx ty::Const<'tcx>, u128>,
    ) -> bool {
        let match_pair = match candidate.match_pairs.iter().find(|mp| mp.place == *test_place) {
            Some(match_pair) => match_pair,
            _ => {
                return false;
            }
        };

        match *match_pair.pattern.kind {
            PatKind::Constant { value } => {
                options.entry(value).or_insert_with(|| {
                    value.eval_bits(self.hir.tcx(), self.hir.param_env, switch_ty)
                });
                true
            }
            PatKind::Variant { .. } => {
                panic!("you should have called add_variants_to_switch instead!");
            }
            PatKind::Range(range) => {
                // Check that none of the switch values are in the range.
                self.values_not_contained_in_range(range, options).unwrap_or(false)
            }
            PatKind::Slice { .. }
            | PatKind::Array { .. }
            | PatKind::Wild
            | PatKind::Or { .. }
            | PatKind::Binding { .. }
            | PatKind::AscribeUserType { .. }
            | PatKind::Leaf { .. }
            | PatKind::Deref { .. } => {
                // don't know how to add these patterns to a switch
                false
            }
        }
    }

    pub(super) fn add_variants_to_switch<'pat>(
        &mut self,
        test_place: &Place<'tcx>,
        candidate: &Candidate<'pat, 'tcx>,
        variants: &mut BitSet<VariantIdx>,
    ) -> bool {
        let match_pair = match candidate.match_pairs.iter().find(|mp| mp.place == *test_place) {
            Some(match_pair) => match_pair,
            _ => {
                return false;
            }
        };

        match *match_pair.pattern.kind {
            PatKind::Variant { adt_def: _, variant_index, .. } => {
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

    pub(super) fn perform_test(
        &mut self,
        block: BasicBlock,
        place: Place<'tcx>,
        test: &Test<'tcx>,
        make_target_blocks: impl FnOnce(&mut Self) -> Vec<BasicBlock>,
    ) {
        debug!(
            "perform_test({:?}, {:?}: {:?}, {:?})",
            block,
            place,
            place.ty(&self.local_decls, self.hir.tcx()),
            test
        );

        let source_info = self.source_info(test.span);
        match test.kind {
            TestKind::Switch { adt_def, ref variants } => {
                let target_blocks = make_target_blocks(self);
                // Variants is a BitVec of indexes into adt_def.variants.
                let num_enum_variants = adt_def.variants.len();
                debug_assert_eq!(target_blocks.len(), num_enum_variants + 1);
                let otherwise_block = *target_blocks.last().unwrap();
                let tcx = self.hir.tcx();
                let switch_targets = SwitchTargets::new(
                    adt_def.discriminants(tcx).filter_map(|(idx, discr)| {
                        if variants.contains(idx) {
                            debug_assert_ne!(
                                target_blocks[idx.index()],
                                otherwise_block,
                                "no canididates for tested discriminant: {:?}",
                                discr,
                            );
                            Some((discr.val, target_blocks[idx.index()]))
                        } else {
                            debug_assert_eq!(
                                target_blocks[idx.index()],
                                otherwise_block,
                                "found canididates for untested discriminant: {:?}",
                                discr,
                            );
                            None
                        }
                    }),
                    otherwise_block,
                );
                debug!("num_enum_variants: {}, variants: {:?}", num_enum_variants, variants);
                let discr_ty = adt_def.repr.discr_type().to_ty(tcx);
                let discr = self.temp(discr_ty, test.span);
                self.cfg.push_assign(block, source_info, discr, Rvalue::Discriminant(place));
                self.cfg.terminate(
                    block,
                    source_info,
                    TerminatorKind::SwitchInt {
                        discr: Operand::Move(discr),
                        switch_ty: discr_ty,
                        targets: switch_targets,
                    },
                );
            }

            TestKind::SwitchInt { switch_ty, ref options } => {
                let target_blocks = make_target_blocks(self);
                let terminator = if *switch_ty.kind() == ty::Bool {
                    assert!(!options.is_empty() && options.len() <= 2);
                    if let [first_bb, second_bb] = *target_blocks {
                        let (true_bb, false_bb) = match options[0] {
                            1 => (first_bb, second_bb),
                            0 => (second_bb, first_bb),
                            v => span_bug!(test.span, "expected boolean value but got {:?}", v),
                        };
                        TerminatorKind::if_(self.hir.tcx(), Operand::Copy(place), true_bb, false_bb)
                    } else {
                        bug!("`TestKind::SwitchInt` on `bool` should have two targets")
                    }
                } else {
                    // The switch may be inexhaustive so we have a catch all block
                    debug_assert_eq!(options.len() + 1, target_blocks.len());
                    let otherwise_block = *target_blocks.last().unwrap();
                    let switch_targets = SwitchTargets::new(
                        options.values().copied().zip(target_blocks),
                        otherwise_block,
                    );
                    TerminatorKind::SwitchInt {
                        discr: Operand::Copy(place),
                        switch_ty,
                        targets: switch_targets,
                    }
                };
                self.cfg.terminate(block, source_info, terminator);
            }

            TestKind::Eq { value, ty } => {
                if !ty.is_scalar() {
                    // Use `PartialEq::eq` instead of `BinOp::Eq`
                    // (the binop can only handle primitives)
                    self.non_scalar_compare(
                        block,
                        make_target_blocks,
                        source_info,
                        value,
                        place,
                        ty,
                    );
                } else if let [success, fail] = *make_target_blocks(self) {
                    assert_eq!(value.ty, ty);
                    let expect = self.literal_operand(test.span, value);
                    let val = Operand::Copy(place);
                    self.compare(block, success, fail, source_info, BinOp::Eq, expect, val);
                } else {
                    bug!("`TestKind::Eq` should have two target blocks");
                }
            }

            TestKind::Range(PatRange { ref lo, ref hi, ref end }) => {
                let lower_bound_success = self.cfg.start_new_block();
                let target_blocks = make_target_blocks(self);

                // Test `val` by computing `lo <= val && val <= hi`, using primitive comparisons.
                let lo = self.literal_operand(test.span, lo);
                let hi = self.literal_operand(test.span, hi);
                let val = Operand::Copy(place);

                if let [success, fail] = *target_blocks {
                    self.compare(
                        block,
                        lower_bound_success,
                        fail,
                        source_info,
                        BinOp::Le,
                        lo,
                        val.clone(),
                    );
                    let op = match *end {
                        RangeEnd::Included => BinOp::Le,
                        RangeEnd::Excluded => BinOp::Lt,
                    };
                    self.compare(lower_bound_success, success, fail, source_info, op, val, hi);
                } else {
                    bug!("`TestKind::Range` should have two target blocks");
                }
            }

            TestKind::Len { len, op } => {
                let target_blocks = make_target_blocks(self);

                let usize_ty = self.hir.usize_ty();
                let actual = self.temp(usize_ty, test.span);

                // actual = len(place)
                self.cfg.push_assign(block, source_info, actual, Rvalue::Len(place));

                // expected = <N>
                let expected = self.push_usize(block, source_info, len);

                if let [true_bb, false_bb] = *target_blocks {
                    // result = actual == expected OR result = actual < expected
                    // branch based on result
                    self.compare(
                        block,
                        true_bb,
                        false_bb,
                        source_info,
                        op,
                        Operand::Move(actual),
                        Operand::Move(expected),
                    );
                } else {
                    bug!("`TestKind::Len` should have two target blocks");
                }
            }
        }
    }

    /// Compare using the provided built-in comparison operator
    fn compare(
        &mut self,
        block: BasicBlock,
        success_block: BasicBlock,
        fail_block: BasicBlock,
        source_info: SourceInfo,
        op: BinOp,
        left: Operand<'tcx>,
        right: Operand<'tcx>,
    ) {
        let bool_ty = self.hir.bool_ty();
        let result = self.temp(bool_ty, source_info.span);

        // result = op(left, right)
        self.cfg.push_assign(block, source_info, result, Rvalue::BinaryOp(op, left, right));

        // branch based on result
        self.cfg.terminate(
            block,
            source_info,
            TerminatorKind::if_(self.hir.tcx(), Operand::Move(result), success_block, fail_block),
        );
    }

    /// Compare two `&T` values using `<T as std::compare::PartialEq>::eq`
    fn non_scalar_compare(
        &mut self,
        block: BasicBlock,
        make_target_blocks: impl FnOnce(&mut Self) -> Vec<BasicBlock>,
        source_info: SourceInfo,
        value: &'tcx ty::Const<'tcx>,
        place: Place<'tcx>,
        mut ty: Ty<'tcx>,
    ) {
        let mut expect = self.literal_operand(source_info.span, value);
        let mut val = Operand::Copy(place);

        // If we're using `b"..."` as a pattern, we need to insert an
        // unsizing coercion, as the byte string has the type `&[u8; N]`.
        //
        // We want to do this even when the scrutinee is a reference to an
        // array, so we can call `<[u8]>::eq` rather than having to find an
        // `<[u8; N]>::eq`.
        let unsize = |ty: Ty<'tcx>| match ty.kind() {
            ty::Ref(region, rty, _) => match rty.kind() {
                ty::Array(inner_ty, n) => Some((region, inner_ty, n)),
                _ => None,
            },
            _ => None,
        };
        let opt_ref_ty = unsize(ty);
        let opt_ref_test_ty = unsize(value.ty);
        match (opt_ref_ty, opt_ref_test_ty) {
            // nothing to do, neither is an array
            (None, None) => {}
            (Some((region, elem_ty, _)), _) | (None, Some((region, elem_ty, _))) => {
                let tcx = self.hir.tcx();
                // make both a slice
                ty = tcx.mk_imm_ref(region, tcx.mk_slice(elem_ty));
                if opt_ref_ty.is_some() {
                    let temp = self.temp(ty, source_info.span);
                    self.cfg.push_assign(
                        block,
                        source_info,
                        temp,
                        Rvalue::Cast(CastKind::Pointer(PointerCast::Unsize), val, ty),
                    );
                    val = Operand::Move(temp);
                }
                if opt_ref_test_ty.is_some() {
                    let slice = self.temp(ty, source_info.span);
                    self.cfg.push_assign(
                        block,
                        source_info,
                        slice,
                        Rvalue::Cast(CastKind::Pointer(PointerCast::Unsize), expect, ty),
                    );
                    expect = Operand::Move(slice);
                }
            }
        }

        let deref_ty = match *ty.kind() {
            ty::Ref(_, deref_ty, _) => deref_ty,
            _ => bug!("non_scalar_compare called on non-reference type: {}", ty),
        };

        let eq_def_id = self.hir.tcx().require_lang_item(LangItem::PartialEq, None);
        let method = self.hir.trait_method(eq_def_id, sym::eq, deref_ty, &[deref_ty.into()]);

        let bool_ty = self.hir.bool_ty();
        let eq_result = self.temp(bool_ty, source_info.span);
        let eq_block = self.cfg.start_new_block();
        self.cfg.terminate(
            block,
            source_info,
            TerminatorKind::Call {
                func: Operand::Constant(box Constant {
                    span: source_info.span,

                    // FIXME(#54571): This constant comes from user input (a
                    // constant in a pattern).  Are there forms where users can add
                    // type annotations here?  For example, an associated constant?
                    // Need to experiment.
                    user_ty: None,

                    literal: method,
                }),
                args: vec![val, expect],
                destination: Some((eq_result, eq_block)),
                cleanup: None,
                from_hir_call: false,
                fn_span: source_info.span,
            },
        );
        self.diverge_from(block);

        if let [success_block, fail_block] = *make_target_blocks(self) {
            // check the result
            self.cfg.terminate(
                eq_block,
                source_info,
                TerminatorKind::if_(
                    self.hir.tcx(),
                    Operand::Move(eq_result),
                    success_block,
                    fail_block,
                ),
            );
        } else {
            bug!("`TestKind::Eq` should have two target blocks")
        }
    }

    /// Given that we are performing `test` against `test_place`, this job
    /// sorts out what the status of `candidate` will be after the test. See
    /// `test_candidates` for the usage of this function. The returned index is
    /// the index that this candidate should be placed in the
    /// `target_candidates` vec. The candidate may be modified to update its
    /// `match_pairs`.
    ///
    /// So, for example, if this candidate is `x @ Some(P0)` and the `Test` is
    /// a variant test, then we would modify the candidate to be `(x as
    /// Option).0 @ P0` and return the index corresponding to the variant
    /// `Some`.
    ///
    /// However, in some cases, the test may just not be relevant to candidate.
    /// For example, suppose we are testing whether `foo.x == 22`, but in one
    /// match arm we have `Foo { x: _, ... }`... in that case, the test for
    /// what value `x` has has no particular relevance to this candidate. In
    /// such cases, this function just returns None without doing anything.
    /// This is used by the overall `match_candidates` algorithm to structure
    /// the match as a whole. See `match_candidates` for more details.
    ///
    /// FIXME(#29623). In some cases, we have some tricky choices to make.  for
    /// example, if we are testing that `x == 22`, but the candidate is `x @
    /// 13..55`, what should we do? In the event that the test is true, we know
    /// that the candidate applies, but in the event of false, we don't know
    /// that it *doesn't* apply. For now, we return false, indicate that the
    /// test does not apply to this candidate, but it might be we can get
    /// tighter match code if we do something a bit different.
    pub(super) fn sort_candidate<'pat>(
        &mut self,
        test_place: &Place<'tcx>,
        test: &Test<'tcx>,
        candidate: &mut Candidate<'pat, 'tcx>,
    ) -> Option<usize> {
        // Find the match_pair for this place (if any). At present,
        // afaik, there can be at most one. (In the future, if we
        // adopted a more general `@` operator, there might be more
        // than one, but it'd be very unusual to have two sides that
        // both require tests; you'd expect one side to be simplified
        // away.)
        let (match_pair_index, match_pair) =
            candidate.match_pairs.iter().enumerate().find(|&(_, mp)| mp.place == *test_place)?;

        match (&test.kind, &*match_pair.pattern.kind) {
            // If we are performing a variant switch, then this
            // informs variant patterns, but nothing else.
            (
                &TestKind::Switch { adt_def: tested_adt_def, .. },
                &PatKind::Variant { adt_def, variant_index, ref subpatterns, .. },
            ) => {
                assert_eq!(adt_def, tested_adt_def);
                self.candidate_after_variant_switch(
                    match_pair_index,
                    adt_def,
                    variant_index,
                    subpatterns,
                    candidate,
                );
                Some(variant_index.as_usize())
            }

            (&TestKind::Switch { .. }, _) => None,

            // If we are performing a switch over integers, then this informs integer
            // equality, but nothing else.
            //
            // FIXME(#29623) we could use PatKind::Range to rule
            // things out here, in some cases.
            (
                &TestKind::SwitchInt { switch_ty: _, ref options },
                &PatKind::Constant { ref value },
            ) if is_switch_ty(match_pair.pattern.ty) => {
                let index = options.get_index_of(value).unwrap();
                self.candidate_without_match_pair(match_pair_index, candidate);
                Some(index)
            }

            (&TestKind::SwitchInt { switch_ty: _, ref options }, &PatKind::Range(range)) => {
                let not_contained =
                    self.values_not_contained_in_range(range, options).unwrap_or(false);

                if not_contained {
                    // No switch values are contained in the pattern range,
                    // so the pattern can be matched only if this test fails.
                    let otherwise = options.len();
                    Some(otherwise)
                } else {
                    None
                }
            }

            (&TestKind::SwitchInt { .. }, _) => None,

            (
                &TestKind::Len { len: test_len, op: BinOp::Eq },
                &PatKind::Slice { ref prefix, ref slice, ref suffix },
            ) => {
                let pat_len = (prefix.len() + suffix.len()) as u64;
                match (test_len.cmp(&pat_len), slice) {
                    (Ordering::Equal, &None) => {
                        // on true, min_len = len = $actual_length,
                        // on false, len != $actual_length
                        self.candidate_after_slice_test(
                            match_pair_index,
                            candidate,
                            prefix,
                            slice.as_ref(),
                            suffix,
                        );
                        Some(0)
                    }
                    (Ordering::Less, _) => {
                        // test_len < pat_len. If $actual_len = test_len,
                        // then $actual_len < pat_len and we don't have
                        // enough elements.
                        Some(1)
                    }
                    (Ordering::Equal | Ordering::Greater, &Some(_)) => {
                        // This can match both if $actual_len = test_len >= pat_len,
                        // and if $actual_len > test_len. We can't advance.
                        None
                    }
                    (Ordering::Greater, &None) => {
                        // test_len != pat_len, so if $actual_len = test_len, then
                        // $actual_len != pat_len.
                        Some(1)
                    }
                }
            }

            (
                &TestKind::Len { len: test_len, op: BinOp::Ge },
                &PatKind::Slice { ref prefix, ref slice, ref suffix },
            ) => {
                // the test is `$actual_len >= test_len`
                let pat_len = (prefix.len() + suffix.len()) as u64;
                match (test_len.cmp(&pat_len), slice) {
                    (Ordering::Equal, &Some(_)) => {
                        // $actual_len >= test_len = pat_len,
                        // so we can match.
                        self.candidate_after_slice_test(
                            match_pair_index,
                            candidate,
                            prefix,
                            slice.as_ref(),
                            suffix,
                        );
                        Some(0)
                    }
                    (Ordering::Less, _) | (Ordering::Equal, &None) => {
                        // test_len <= pat_len. If $actual_len < test_len,
                        // then it is also < pat_len, so the test passing is
                        // necessary (but insufficient).
                        Some(0)
                    }
                    (Ordering::Greater, &None) => {
                        // test_len > pat_len. If $actual_len >= test_len > pat_len,
                        // then we know we won't have a match.
                        Some(1)
                    }
                    (Ordering::Greater, &Some(_)) => {
                        // test_len < pat_len, and is therefore less
                        // strict. This can still go both ways.
                        None
                    }
                }
            }

            (&TestKind::Range(test), &PatKind::Range(pat)) => {
                if test == pat {
                    self.candidate_without_match_pair(match_pair_index, candidate);
                    return Some(0);
                }

                let no_overlap = (|| {
                    use rustc_hir::RangeEnd::*;
                    use std::cmp::Ordering::*;

                    let tcx = self.hir.tcx();

                    let test_ty = test.lo.ty;
                    let lo = compare_const_vals(tcx, test.lo, pat.hi, self.hir.param_env, test_ty)?;
                    let hi = compare_const_vals(tcx, test.hi, pat.lo, self.hir.param_env, test_ty)?;

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

                if let Some(true) = no_overlap {
                    // Testing range does not overlap with pattern range,
                    // so the pattern can be matched only if this test fails.
                    Some(1)
                } else {
                    None
                }
            }

            (&TestKind::Range(range), &PatKind::Constant { value }) => {
                if let Some(false) = self.const_range_contains(range, value) {
                    // `value` is not contained in the testing range,
                    // so `value` can be matched only if this test fails.
                    Some(1)
                } else {
                    None
                }
            }

            (&TestKind::Range { .. }, _) => None,

            (&TestKind::Eq { .. } | &TestKind::Len { .. }, _) => {
                // These are all binary tests.
                //
                // FIXME(#29623) we can be more clever here
                let pattern_test = self.test(&match_pair);
                if pattern_test.kind == test.kind {
                    self.candidate_without_match_pair(match_pair_index, candidate);
                    Some(0)
                } else {
                    None
                }
            }
        }
    }

    fn candidate_without_match_pair(
        &mut self,
        match_pair_index: usize,
        candidate: &mut Candidate<'_, 'tcx>,
    ) {
        candidate.match_pairs.remove(match_pair_index);
    }

    fn candidate_after_slice_test<'pat>(
        &mut self,
        match_pair_index: usize,
        candidate: &mut Candidate<'pat, 'tcx>,
        prefix: &'pat [Pat<'tcx>],
        opt_slice: Option<&'pat Pat<'tcx>>,
        suffix: &'pat [Pat<'tcx>],
    ) {
        let removed_place = candidate.match_pairs.remove(match_pair_index).place;
        self.prefix_slice_suffix(
            &mut candidate.match_pairs,
            &removed_place,
            prefix,
            opt_slice,
            suffix,
        );
    }

    fn candidate_after_variant_switch<'pat>(
        &mut self,
        match_pair_index: usize,
        adt_def: &'tcx ty::AdtDef,
        variant_index: VariantIdx,
        subpatterns: &'pat [FieldPat<'tcx>],
        candidate: &mut Candidate<'pat, 'tcx>,
    ) {
        let match_pair = candidate.match_pairs.remove(match_pair_index);
        let tcx = self.hir.tcx();

        // So, if we have a match-pattern like `x @ Enum::Variant(P1, P2)`,
        // we want to create a set of derived match-patterns like
        // `(x as Variant).0 @ P1` and `(x as Variant).1 @ P1`.
        let elem = ProjectionElem::Downcast(
            Some(adt_def.variants[variant_index].ident.name),
            variant_index,
        );
        let downcast_place = tcx.mk_place_elem(match_pair.place, elem); // `(x as Variant)`
        let consequent_match_pairs = subpatterns.iter().map(|subpattern| {
            // e.g., `(x as Variant).0`
            let place = tcx.mk_place_field(downcast_place, subpattern.field, subpattern.pattern.ty);
            // e.g., `(x as Variant).0 @ P1`
            MatchPair::new(place, &subpattern.pattern)
        });

        candidate.match_pairs.extend(consequent_match_pairs);
    }

    fn error_simplifyable<'pat>(&mut self, match_pair: &MatchPair<'pat, 'tcx>) -> ! {
        span_bug!(match_pair.pattern.span, "simplifyable pattern found: {:?}", match_pair.pattern)
    }

    fn const_range_contains(
        &self,
        range: PatRange<'tcx>,
        value: &'tcx ty::Const<'tcx>,
    ) -> Option<bool> {
        use std::cmp::Ordering::*;

        let tcx = self.hir.tcx();

        let a = compare_const_vals(tcx, range.lo, value, self.hir.param_env, range.lo.ty)?;
        let b = compare_const_vals(tcx, value, range.hi, self.hir.param_env, range.lo.ty)?;

        match (b, range.end) {
            (Less, _) | (Equal, RangeEnd::Included) if a != Greater => Some(true),
            _ => Some(false),
        }
    }

    fn values_not_contained_in_range(
        &self,
        range: PatRange<'tcx>,
        options: &FxIndexMap<&'tcx ty::Const<'tcx>, u128>,
    ) -> Option<bool> {
        for &val in options.keys() {
            if self.const_range_contains(range, val)? {
                return Some(false);
            }
        }

        Some(true)
    }
}

impl Test<'_> {
    pub(super) fn targets(&self) -> usize {
        match self.kind {
            TestKind::Eq { .. } | TestKind::Range(_) | TestKind::Len { .. } => 2,
            TestKind::Switch { adt_def, .. } => {
                // While the switch that we generate doesn't test for all
                // variants, we have a target for each variant and the
                // otherwise case, and we make sure that all of the cases not
                // specified have the same block.
                adt_def.variants.len() + 1
            }
            TestKind::SwitchInt { switch_ty, ref options, .. } => {
                if switch_ty.is_bool() {
                    // `bool` is special cased in `perform_test` to always
                    // branch to two blocks.
                    2
                } else {
                    options.len() + 1
                }
            }
        }
    }
}

fn is_switch_ty(ty: Ty<'_>) -> bool {
    ty.is_integral() || ty.is_char() || ty.is_bool()
}
