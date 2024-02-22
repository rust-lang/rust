// Testing candidates
//
// After candidates have been simplified, the only match pairs that
// remain are those that require some sort of test. The functions here
// identify what tests are needed, perform the tests, and then filter
// the candidates based on the result.

use crate::build::expr::as_place::PlaceBuilder;
use crate::build::matches::{Candidate, MatchPair, Test, TestKind};
use crate::build::Builder;
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::{LangItem, RangeEnd};
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::*;
use rustc_middle::thir::*;
use rustc_middle::ty::util::IntTypeExt;
use rustc_middle::ty::GenericArg;
use rustc_middle::ty::{self, adjustment::PointerCoercion, Ty, TyCtxt};
use rustc_span::def_id::DefId;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::VariantIdx;

use std::cmp::Ordering;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Identifies what test is needed to decide if `match_pair` is applicable.
    ///
    /// It is a bug to call this with a not-fully-simplified pattern.
    pub(super) fn test<'pat>(&mut self, match_pair: &MatchPair<'pat, 'tcx>) -> Test<'tcx> {
        match match_pair.pattern.kind {
            PatKind::Variant { adt_def, args: _, variant_index: _, subpatterns: _ } => Test {
                span: match_pair.pattern.span,
                kind: TestKind::Switch {
                    adt_def,
                    variants: BitSet::new_empty(adt_def.variants().len()),
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
                kind: TestKind::Eq { value, ty: match_pair.pattern.ty },
            },

            PatKind::Range(ref range) => {
                assert_eq!(range.ty, match_pair.pattern.ty);
                Test { span: match_pair.pattern.span, kind: TestKind::Range(range.clone()) }
            }

            PatKind::Slice { ref prefix, ref slice, ref suffix } => {
                let len = prefix.len() + suffix.len();
                let op = if slice.is_some() { BinOp::Ge } else { BinOp::Eq };
                Test { span: match_pair.pattern.span, kind: TestKind::Len { len: len as u64, op } }
            }

            PatKind::Or { .. } => bug!("or-patterns should have already been handled"),

            PatKind::AscribeUserType { .. }
            | PatKind::InlineConstant { .. }
            | PatKind::Array { .. }
            | PatKind::Wild
            | PatKind::Binding { .. }
            | PatKind::Never
            | PatKind::Leaf { .. }
            | PatKind::Deref { .. }
            | PatKind::Error(_) => self.error_simplifiable(match_pair),
        }
    }

    pub(super) fn add_cases_to_switch<'pat>(
        &mut self,
        test_place: &PlaceBuilder<'tcx>,
        candidate: &Candidate<'pat, 'tcx>,
        options: &mut FxIndexMap<Const<'tcx>, u128>,
    ) -> bool {
        let Some(match_pair) = candidate.match_pairs.iter().find(|mp| mp.place == *test_place)
        else {
            return false;
        };

        match match_pair.pattern.kind {
            PatKind::Constant { value } => {
                options.entry(value).or_insert_with(|| value.eval_bits(self.tcx, self.param_env));
                true
            }
            PatKind::Variant { .. } => {
                panic!("you should have called add_variants_to_switch instead!");
            }
            PatKind::Range(ref range) => {
                // Check that none of the switch values are in the range.
                self.values_not_contained_in_range(&*range, options).unwrap_or(false)
            }
            PatKind::Slice { .. }
            | PatKind::Array { .. }
            | PatKind::Wild
            | PatKind::Never
            | PatKind::Or { .. }
            | PatKind::Binding { .. }
            | PatKind::AscribeUserType { .. }
            | PatKind::InlineConstant { .. }
            | PatKind::Leaf { .. }
            | PatKind::Deref { .. }
            | PatKind::Error(_) => {
                // don't know how to add these patterns to a switch
                false
            }
        }
    }

    pub(super) fn add_variants_to_switch<'pat>(
        &mut self,
        test_place: &PlaceBuilder<'tcx>,
        candidate: &Candidate<'pat, 'tcx>,
        variants: &mut BitSet<VariantIdx>,
    ) -> bool {
        let Some(match_pair) = candidate.match_pairs.iter().find(|mp| mp.place == *test_place)
        else {
            return false;
        };

        match match_pair.pattern.kind {
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

    #[instrument(skip(self, target_blocks, place_builder), level = "debug")]
    pub(super) fn perform_test(
        &mut self,
        match_start_span: Span,
        scrutinee_span: Span,
        block: BasicBlock,
        place_builder: &PlaceBuilder<'tcx>,
        test: &Test<'tcx>,
        target_blocks: Vec<BasicBlock>,
    ) {
        let place = place_builder.to_place(self);
        let place_ty = place.ty(&self.local_decls, self.tcx);
        debug!(?place, ?place_ty,);

        let source_info = self.source_info(test.span);
        match test.kind {
            TestKind::Switch { adt_def, ref variants } => {
                // Variants is a BitVec of indexes into adt_def.variants.
                let num_enum_variants = adt_def.variants().len();
                debug_assert_eq!(target_blocks.len(), num_enum_variants + 1);
                let otherwise_block = *target_blocks.last().unwrap();
                let tcx = self.tcx;
                let switch_targets = SwitchTargets::new(
                    adt_def.discriminants(tcx).filter_map(|(idx, discr)| {
                        if variants.contains(idx) {
                            debug_assert_ne!(
                                target_blocks[idx.index()],
                                otherwise_block,
                                "no candidates for tested discriminant: {discr:?}",
                            );
                            Some((discr.val, target_blocks[idx.index()]))
                        } else {
                            debug_assert_eq!(
                                target_blocks[idx.index()],
                                otherwise_block,
                                "found candidates for untested discriminant: {discr:?}",
                            );
                            None
                        }
                    }),
                    otherwise_block,
                );
                debug!("num_enum_variants: {}, variants: {:?}", num_enum_variants, variants);
                let discr_ty = adt_def.repr().discr_type().to_ty(tcx);
                let discr = self.temp(discr_ty, test.span);
                self.cfg.push_assign(
                    block,
                    self.source_info(scrutinee_span),
                    discr,
                    Rvalue::Discriminant(place),
                );
                self.cfg.terminate(
                    block,
                    self.source_info(match_start_span),
                    TerminatorKind::SwitchInt {
                        discr: Operand::Move(discr),
                        targets: switch_targets,
                    },
                );
            }

            TestKind::SwitchInt { switch_ty, ref options } => {
                let terminator = if *switch_ty.kind() == ty::Bool {
                    assert!(!options.is_empty() && options.len() <= 2);
                    let [first_bb, second_bb] = *target_blocks else {
                        bug!("`TestKind::SwitchInt` on `bool` should have two targets")
                    };
                    let (true_bb, false_bb) = match options[0] {
                        1 => (first_bb, second_bb),
                        0 => (second_bb, first_bb),
                        v => span_bug!(test.span, "expected boolean value but got {:?}", v),
                    };
                    TerminatorKind::if_(Operand::Copy(place), true_bb, false_bb)
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
                        targets: switch_targets,
                    }
                };
                self.cfg.terminate(block, self.source_info(match_start_span), terminator);
            }

            TestKind::Eq { value, ty } => {
                let tcx = self.tcx;
                let [success_block, fail_block] = *target_blocks else {
                    bug!("`TestKind::Eq` should have two target blocks")
                };
                if let ty::Adt(def, _) = ty.kind()
                    && Some(def.did()) == tcx.lang_items().string()
                {
                    if !tcx.features().string_deref_patterns {
                        bug!(
                            "matching on `String` went through without enabling string_deref_patterns"
                        );
                    }
                    let re_erased = tcx.lifetimes.re_erased;
                    let ref_string = self.temp(Ty::new_imm_ref(tcx, re_erased, ty), test.span);
                    let ref_str_ty = Ty::new_imm_ref(tcx, re_erased, tcx.types.str_);
                    let ref_str = self.temp(ref_str_ty, test.span);
                    let deref = tcx.require_lang_item(LangItem::Deref, None);
                    let method = trait_method(tcx, deref, sym::deref, [ty]);
                    let eq_block = self.cfg.start_new_block();
                    self.cfg.push_assign(
                        block,
                        source_info,
                        ref_string,
                        Rvalue::Ref(re_erased, BorrowKind::Shared, place),
                    );
                    self.cfg.terminate(
                        block,
                        source_info,
                        TerminatorKind::Call {
                            func: Operand::Constant(Box::new(ConstOperand {
                                span: test.span,
                                user_ty: None,
                                const_: method,
                            })),
                            args: vec![Spanned { node: Operand::Move(ref_string), span: DUMMY_SP }],
                            destination: ref_str,
                            target: Some(eq_block),
                            unwind: UnwindAction::Continue,
                            call_source: CallSource::Misc,
                            fn_span: source_info.span,
                        },
                    );
                    self.non_scalar_compare(
                        eq_block,
                        success_block,
                        fail_block,
                        source_info,
                        value,
                        ref_str,
                        ref_str_ty,
                    );
                } else if !ty.is_scalar() {
                    // Use `PartialEq::eq` instead of `BinOp::Eq`
                    // (the binop can only handle primitives)
                    self.non_scalar_compare(
                        block,
                        success_block,
                        fail_block,
                        source_info,
                        value,
                        place,
                        ty,
                    );
                } else {
                    assert_eq!(value.ty(), ty);
                    let expect = self.literal_operand(test.span, value);
                    let val = Operand::Copy(place);
                    self.compare(
                        block,
                        success_block,
                        fail_block,
                        source_info,
                        BinOp::Eq,
                        expect,
                        val,
                    );
                }
            }

            TestKind::Range(ref range) => {
                let lower_bound_success = self.cfg.start_new_block();

                // Test `val` by computing `lo <= val && val <= hi`, using primitive comparisons.
                // FIXME: skip useless comparison when the range is half-open.
                let lo = range.lo.to_const(range.ty, self.tcx);
                let hi = range.hi.to_const(range.ty, self.tcx);
                let lo = self.literal_operand(test.span, lo);
                let hi = self.literal_operand(test.span, hi);
                let val = Operand::Copy(place);

                let [success, fail] = *target_blocks else {
                    bug!("`TestKind::Range` should have two target blocks");
                };
                self.compare(
                    block,
                    lower_bound_success,
                    fail,
                    source_info,
                    BinOp::Le,
                    lo,
                    val.clone(),
                );
                let op = match range.end {
                    RangeEnd::Included => BinOp::Le,
                    RangeEnd::Excluded => BinOp::Lt,
                };
                self.compare(lower_bound_success, success, fail, source_info, op, val, hi);
            }

            TestKind::Len { len, op } => {
                let usize_ty = self.tcx.types.usize;
                let actual = self.temp(usize_ty, test.span);

                // actual = len(place)
                self.cfg.push_assign(block, source_info, actual, Rvalue::Len(place));

                // expected = <N>
                let expected = self.push_usize(block, source_info, len);

                let [true_bb, false_bb] = *target_blocks else {
                    bug!("`TestKind::Len` should have two target blocks");
                };
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
        let bool_ty = self.tcx.types.bool;
        let result = self.temp(bool_ty, source_info.span);

        // result = op(left, right)
        self.cfg.push_assign(
            block,
            source_info,
            result,
            Rvalue::BinaryOp(op, Box::new((left, right))),
        );

        // branch based on result
        self.cfg.terminate(
            block,
            source_info,
            TerminatorKind::if_(Operand::Move(result), success_block, fail_block),
        );
    }

    /// Compare two values using `<T as std::compare::PartialEq>::eq`.
    /// If the values are already references, just call it directly, otherwise
    /// take a reference to the values first and then call it.
    fn non_scalar_compare(
        &mut self,
        block: BasicBlock,
        success_block: BasicBlock,
        fail_block: BasicBlock,
        source_info: SourceInfo,
        value: Const<'tcx>,
        mut val: Place<'tcx>,
        mut ty: Ty<'tcx>,
    ) {
        let mut expect = self.literal_operand(source_info.span, value);

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
        let opt_ref_test_ty = unsize(value.ty());
        match (opt_ref_ty, opt_ref_test_ty) {
            // nothing to do, neither is an array
            (None, None) => {}
            (Some((region, elem_ty, _)), _) | (None, Some((region, elem_ty, _))) => {
                let tcx = self.tcx;
                // make both a slice
                ty = Ty::new_imm_ref(tcx, *region, Ty::new_slice(tcx, *elem_ty));
                if opt_ref_ty.is_some() {
                    let temp = self.temp(ty, source_info.span);
                    self.cfg.push_assign(
                        block,
                        source_info,
                        temp,
                        Rvalue::Cast(
                            CastKind::PointerCoercion(PointerCoercion::Unsize),
                            Operand::Copy(val),
                            ty,
                        ),
                    );
                    val = temp;
                }
                if opt_ref_test_ty.is_some() {
                    let slice = self.temp(ty, source_info.span);
                    self.cfg.push_assign(
                        block,
                        source_info,
                        slice,
                        Rvalue::Cast(
                            CastKind::PointerCoercion(PointerCoercion::Unsize),
                            expect,
                            ty,
                        ),
                    );
                    expect = Operand::Move(slice);
                }
            }
        }

        match *ty.kind() {
            ty::Ref(_, deref_ty, _) => ty = deref_ty,
            _ => {
                // non_scalar_compare called on non-reference type
                let temp = self.temp(ty, source_info.span);
                self.cfg.push_assign(block, source_info, temp, Rvalue::Use(expect));
                let ref_ty = Ty::new_imm_ref(self.tcx, self.tcx.lifetimes.re_erased, ty);
                let ref_temp = self.temp(ref_ty, source_info.span);

                self.cfg.push_assign(
                    block,
                    source_info,
                    ref_temp,
                    Rvalue::Ref(self.tcx.lifetimes.re_erased, BorrowKind::Shared, temp),
                );
                expect = Operand::Move(ref_temp);

                let ref_temp = self.temp(ref_ty, source_info.span);
                self.cfg.push_assign(
                    block,
                    source_info,
                    ref_temp,
                    Rvalue::Ref(self.tcx.lifetimes.re_erased, BorrowKind::Shared, val),
                );
                val = ref_temp;
            }
        }

        let eq_def_id = self.tcx.require_lang_item(LangItem::PartialEq, Some(source_info.span));
        let method = trait_method(
            self.tcx,
            eq_def_id,
            sym::eq,
            self.tcx.with_opt_host_effect_param(self.def_id, eq_def_id, [ty, ty]),
        );

        let bool_ty = self.tcx.types.bool;
        let eq_result = self.temp(bool_ty, source_info.span);
        let eq_block = self.cfg.start_new_block();
        self.cfg.terminate(
            block,
            source_info,
            TerminatorKind::Call {
                func: Operand::Constant(Box::new(ConstOperand {
                    span: source_info.span,

                    // FIXME(#54571): This constant comes from user input (a
                    // constant in a pattern). Are there forms where users can add
                    // type annotations here?  For example, an associated constant?
                    // Need to experiment.
                    user_ty: None,

                    const_: method,
                })),
                args: vec![
                    Spanned { node: Operand::Copy(val), span: DUMMY_SP },
                    Spanned { node: expect, span: DUMMY_SP },
                ],
                destination: eq_result,
                target: Some(eq_block),
                unwind: UnwindAction::Continue,
                call_source: CallSource::MatchCmp,
                fn_span: source_info.span,
            },
        );
        self.diverge_from(block);

        // check the result
        self.cfg.terminate(
            eq_block,
            source_info,
            TerminatorKind::if_(Operand::Move(eq_result), success_block, fail_block),
        );
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
    pub(super) fn sort_candidate<'pat>(
        &mut self,
        test_place: &PlaceBuilder<'tcx>,
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

        let fully_matched;
        let ret = match (&test.kind, &match_pair.pattern.kind) {
            // If we are performing a variant switch, then this
            // informs variant patterns, but nothing else.
            (
                &TestKind::Switch { adt_def: tested_adt_def, .. },
                &PatKind::Variant { adt_def, variant_index, .. },
            ) => {
                assert_eq!(adt_def, tested_adt_def);
                fully_matched = true;
                Some(variant_index.as_usize())
            }
            (&TestKind::Switch { .. }, _) => {
                fully_matched = false;
                None
            }

            // If we are performing a switch over integers, then this informs integer
            // equality, but nothing else.
            //
            // FIXME(#29623) we could use PatKind::Range to rule
            // things out here, in some cases.
            (TestKind::SwitchInt { switch_ty: _, options }, PatKind::Constant { value })
                if is_switch_ty(match_pair.pattern.ty) =>
            {
                fully_matched = true;
                let index = options.get_index_of(value).unwrap();
                Some(index)
            }
            (TestKind::SwitchInt { switch_ty: _, options }, PatKind::Range(range)) => {
                fully_matched = false;
                let not_contained =
                    self.values_not_contained_in_range(&*range, options).unwrap_or(false);

                not_contained.then(|| {
                    // No switch values are contained in the pattern range,
                    // so the pattern can be matched only if this test fails.
                    options.len()
                })
            }
            (&TestKind::SwitchInt { .. }, _) => {
                fully_matched = false;
                None
            }

            (
                &TestKind::Len { len: test_len, op: BinOp::Eq },
                PatKind::Slice { prefix, slice, suffix },
            ) => {
                let pat_len = (prefix.len() + suffix.len()) as u64;
                match (test_len.cmp(&pat_len), slice) {
                    (Ordering::Equal, &None) => {
                        // on true, min_len = len = $actual_length,
                        // on false, len != $actual_length
                        fully_matched = true;
                        Some(0)
                    }
                    (Ordering::Less, _) => {
                        // test_len < pat_len. If $actual_len = test_len,
                        // then $actual_len < pat_len and we don't have
                        // enough elements.
                        fully_matched = false;
                        Some(1)
                    }
                    (Ordering::Equal | Ordering::Greater, &Some(_)) => {
                        // This can match both if $actual_len = test_len >= pat_len,
                        // and if $actual_len > test_len. We can't advance.
                        fully_matched = false;
                        None
                    }
                    (Ordering::Greater, &None) => {
                        // test_len != pat_len, so if $actual_len = test_len, then
                        // $actual_len != pat_len.
                        fully_matched = false;
                        Some(1)
                    }
                }
            }
            (
                &TestKind::Len { len: test_len, op: BinOp::Ge },
                PatKind::Slice { prefix, slice, suffix },
            ) => {
                // the test is `$actual_len >= test_len`
                let pat_len = (prefix.len() + suffix.len()) as u64;
                match (test_len.cmp(&pat_len), slice) {
                    (Ordering::Equal, &Some(_)) => {
                        // $actual_len >= test_len = pat_len,
                        // so we can match.
                        fully_matched = true;
                        Some(0)
                    }
                    (Ordering::Less, _) | (Ordering::Equal, &None) => {
                        // test_len <= pat_len. If $actual_len < test_len,
                        // then it is also < pat_len, so the test passing is
                        // necessary (but insufficient).
                        fully_matched = false;
                        Some(0)
                    }
                    (Ordering::Greater, &None) => {
                        // test_len > pat_len. If $actual_len >= test_len > pat_len,
                        // then we know we won't have a match.
                        fully_matched = false;
                        Some(1)
                    }
                    (Ordering::Greater, &Some(_)) => {
                        // test_len < pat_len, and is therefore less
                        // strict. This can still go both ways.
                        fully_matched = false;
                        None
                    }
                }
            }

            (TestKind::Range(test), PatKind::Range(pat)) => {
                if test == pat {
                    fully_matched = true;
                    Some(0)
                } else {
                    fully_matched = false;
                    // If the testing range does not overlap with pattern range,
                    // the pattern can be matched only if this test fails.
                    if !test.overlaps(pat, self.tcx, self.param_env)? { Some(1) } else { None }
                }
            }
            (TestKind::Range(range), &PatKind::Constant { value }) => {
                fully_matched = false;
                if !range.contains(value, self.tcx, self.param_env)? {
                    // `value` is not contained in the testing range,
                    // so `value` can be matched only if this test fails.
                    Some(1)
                } else {
                    None
                }
            }
            (&TestKind::Range { .. }, _) => {
                fully_matched = false;
                None
            }

            (&TestKind::Eq { .. } | &TestKind::Len { .. }, _) => {
                // The call to `self.test(&match_pair)` below is not actually used to generate any
                // MIR. Instead, we just want to compare with `test` (the parameter of the method)
                // to see if it is the same.
                //
                // However, at this point we can still encounter or-patterns that were extracted
                // from previous calls to `sort_candidate`, so we need to manually address that
                // case to avoid panicking in `self.test()`.
                if let PatKind::Or { .. } = &match_pair.pattern.kind {
                    return None;
                }

                // These are all binary tests.
                //
                // FIXME(#29623) we can be more clever here
                let pattern_test = self.test(match_pair);
                if pattern_test.kind == test.kind {
                    fully_matched = true;
                    Some(0)
                } else {
                    fully_matched = false;
                    None
                }
            }
        };

        if fully_matched {
            // Replace the match pair by its sub-pairs.
            let match_pair = candidate.match_pairs.remove(match_pair_index);
            candidate.match_pairs.extend(match_pair.subpairs);
            // Move or-patterns to the end.
            candidate
                .match_pairs
                .sort_by_key(|pair| matches!(pair.pattern.kind, PatKind::Or { .. }));
        }

        ret
    }

    fn error_simplifiable<'pat>(&mut self, match_pair: &MatchPair<'pat, 'tcx>) -> ! {
        span_bug!(match_pair.pattern.span, "simplifiable pattern found: {:?}", match_pair.pattern)
    }

    fn values_not_contained_in_range(
        &self,
        range: &PatRange<'tcx>,
        options: &FxIndexMap<Const<'tcx>, u128>,
    ) -> Option<bool> {
        for &val in options.keys() {
            if range.contains(val, self.tcx, self.param_env)? {
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
                adt_def.variants().len() + 1
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

fn trait_method<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_def_id: DefId,
    method_name: Symbol,
    args: impl IntoIterator<Item: Into<GenericArg<'tcx>>>,
) -> Const<'tcx> {
    // The unhygienic comparison here is acceptable because this is only
    // used on known traits.
    let item = tcx
        .associated_items(trait_def_id)
        .filter_by_name_unhygienic(method_name)
        .find(|item| item.kind == ty::AssocKind::Fn)
        .expect("trait method not found");

    let method_ty = Ty::new_fn_def(tcx, item.def_id, args);

    Const::zero_sized(method_ty)
}
