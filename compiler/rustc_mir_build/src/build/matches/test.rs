// Testing candidates
//
// After candidates have been simplified, the only match pairs that
// remain are those that require some sort of test. The functions here
// identify what tests are needed, perform the tests, and then filter
// the candidates based on the result.

use crate::build::matches::{Candidate, MatchPairTree, Test, TestBranch, TestCase, TestKind};
use crate::build::Builder;
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::{LangItem, RangeEnd};
use rustc_middle::mir::*;
use rustc_middle::ty::util::IntTypeExt;
use rustc_middle::ty::GenericArg;
use rustc_middle::ty::{self, adjustment::PointerCoercion, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_span::def_id::DefId;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{Span, DUMMY_SP};
use tracing::{debug, instrument};

use std::cmp::Ordering;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Identifies what test is needed to decide if `match_pair` is applicable.
    ///
    /// It is a bug to call this with a not-fully-simplified pattern.
    pub(super) fn pick_test_for_match_pair<'pat>(
        &mut self,
        match_pair: &MatchPairTree<'pat, 'tcx>,
    ) -> Test<'tcx> {
        let kind = match match_pair.test_case {
            TestCase::Variant { adt_def, variant_index: _ } => TestKind::Switch { adt_def },

            TestCase::Constant { .. } if match_pair.pattern.ty.is_bool() => TestKind::If,
            TestCase::Constant { .. } if is_switch_ty(match_pair.pattern.ty) => TestKind::SwitchInt,
            TestCase::Constant { value } => TestKind::Eq { value, ty: match_pair.pattern.ty },

            TestCase::Range(range) => {
                assert_eq!(range.ty, match_pair.pattern.ty);
                TestKind::Range(Box::new(range.clone()))
            }

            TestCase::Slice { len, variable_length } => {
                let op = if variable_length { BinOp::Ge } else { BinOp::Eq };
                TestKind::Len { len: len as u64, op }
            }

            TestCase::Deref { temp, mutability } => TestKind::Deref { temp, mutability },

            TestCase::Never => TestKind::Never,

            // Or-patterns are not tested directly; instead they are expanded into subcandidates,
            // which are then distinguished by testing whatever non-or patterns they contain.
            TestCase::Or { .. } => bug!("or-patterns should have already been handled"),

            TestCase::Irrefutable { .. } => span_bug!(
                match_pair.pattern.span,
                "simplifiable pattern found: {:?}",
                match_pair.pattern
            ),
        };

        Test { span: match_pair.pattern.span, kind }
    }

    #[instrument(skip(self, target_blocks, place), level = "debug")]
    pub(super) fn perform_test(
        &mut self,
        match_start_span: Span,
        scrutinee_span: Span,
        block: BasicBlock,
        otherwise_block: BasicBlock,
        place: Place<'tcx>,
        test: &Test<'tcx>,
        target_blocks: FxIndexMap<TestBranch<'tcx>, BasicBlock>,
    ) {
        let place_ty = place.ty(&self.local_decls, self.tcx);
        debug!(?place, ?place_ty);
        let target_block = |branch| target_blocks.get(&branch).copied().unwrap_or(otherwise_block);

        let source_info = self.source_info(test.span);
        match test.kind {
            TestKind::Switch { adt_def } => {
                let otherwise_block = target_block(TestBranch::Failure);
                let switch_targets = SwitchTargets::new(
                    adt_def.discriminants(self.tcx).filter_map(|(idx, discr)| {
                        if let Some(&block) = target_blocks.get(&TestBranch::Variant(idx)) {
                            Some((discr.val, block))
                        } else {
                            None
                        }
                    }),
                    otherwise_block,
                );
                debug!("num_enum_variants: {}", adt_def.variants().len());
                let discr_ty = adt_def.repr().discr_type().to_ty(self.tcx);
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

            TestKind::SwitchInt => {
                // The switch may be inexhaustive so we have a catch-all block
                let otherwise_block = target_block(TestBranch::Failure);
                let switch_targets = SwitchTargets::new(
                    target_blocks.iter().filter_map(|(&branch, &block)| {
                        if let TestBranch::Constant(_, bits) = branch {
                            Some((bits, block))
                        } else {
                            None
                        }
                    }),
                    otherwise_block,
                );
                let terminator = TerminatorKind::SwitchInt {
                    discr: Operand::Copy(place),
                    targets: switch_targets,
                };
                self.cfg.terminate(block, self.source_info(match_start_span), terminator);
            }

            TestKind::If => {
                let success_block = target_block(TestBranch::Success);
                let fail_block = target_block(TestBranch::Failure);
                let terminator =
                    TerminatorKind::if_(Operand::Copy(place), success_block, fail_block);
                self.cfg.terminate(block, self.source_info(match_start_span), terminator);
            }

            TestKind::Eq { value, ty } => {
                let tcx = self.tcx;
                let success_block = target_block(TestBranch::Success);
                let fail_block = target_block(TestBranch::Failure);
                if let ty::Adt(def, _) = ty.kind()
                    && tcx.is_lang_item(def.did(), LangItem::String)
                {
                    if !tcx.features().string_deref_patterns {
                        span_bug!(
                            test.span,
                            "matching on `String` went through without enabling string_deref_patterns"
                        );
                    }
                    let re_erased = tcx.lifetimes.re_erased;
                    let ref_str_ty = Ty::new_imm_ref(tcx, re_erased, tcx.types.str_);
                    let ref_str = self.temp(ref_str_ty, test.span);
                    let eq_block = self.cfg.start_new_block();
                    // `let ref_str: &str = <String as Deref>::deref(&place);`
                    self.call_deref(
                        block,
                        eq_block,
                        place,
                        Mutability::Not,
                        ty,
                        ref_str,
                        test.span,
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
                let success = target_block(TestBranch::Success);
                let fail = target_block(TestBranch::Failure);
                // Test `val` by computing `lo <= val && val <= hi`, using primitive comparisons.
                let val = Operand::Copy(place);

                let intermediate_block = if !range.lo.is_finite() {
                    block
                } else if !range.hi.is_finite() {
                    success
                } else {
                    self.cfg.start_new_block()
                };

                if let Some(lo) = range.lo.as_finite() {
                    let lo = self.literal_operand(test.span, lo);
                    self.compare(
                        block,
                        intermediate_block,
                        fail,
                        source_info,
                        BinOp::Le,
                        lo,
                        val.clone(),
                    );
                };

                if let Some(hi) = range.hi.as_finite() {
                    let hi = self.literal_operand(test.span, hi);
                    let op = match range.end {
                        RangeEnd::Included => BinOp::Le,
                        RangeEnd::Excluded => BinOp::Lt,
                    };
                    self.compare(intermediate_block, success, fail, source_info, op, val, hi);
                }
            }

            TestKind::Len { len, op } => {
                let usize_ty = self.tcx.types.usize;
                let actual = self.temp(usize_ty, test.span);

                // actual = len(place)
                self.cfg.push_assign(block, source_info, actual, Rvalue::Len(place));

                // expected = <N>
                let expected = self.push_usize(block, source_info, len);

                let success_block = target_block(TestBranch::Success);
                let fail_block = target_block(TestBranch::Failure);
                // result = actual == expected OR result = actual < expected
                // branch based on result
                self.compare(
                    block,
                    success_block,
                    fail_block,
                    source_info,
                    op,
                    Operand::Move(actual),
                    Operand::Move(expected),
                );
            }

            TestKind::Deref { temp, mutability } => {
                let ty = place_ty.ty;
                let target = target_block(TestBranch::Success);
                self.call_deref(block, target, place, mutability, ty, temp, test.span);
            }

            TestKind::Never => {
                // Check that the place is initialized.
                // FIXME(never_patterns): Also assert validity of the data at `place`.
                self.cfg.push_fake_read(
                    block,
                    source_info,
                    FakeReadCause::ForMatchedPlace(None),
                    place,
                );
                // A never pattern is only allowed on an uninhabited type, so validity of the data
                // implies unreachability.
                self.cfg.terminate(block, source_info, TerminatorKind::Unreachable);
            }
        }
    }

    /// Perform `let temp = <ty as Deref>::deref(&place)`.
    /// or `let temp = <ty as DerefMut>::deref_mut(&mut place)`.
    pub(super) fn call_deref(
        &mut self,
        block: BasicBlock,
        target_block: BasicBlock,
        place: Place<'tcx>,
        mutability: Mutability,
        ty: Ty<'tcx>,
        temp: Place<'tcx>,
        span: Span,
    ) {
        let (trait_item, method) = match mutability {
            Mutability::Not => (LangItem::Deref, sym::deref),
            Mutability::Mut => (LangItem::DerefMut, sym::deref_mut),
        };
        let borrow_kind = super::util::ref_pat_borrow_kind(mutability);
        let source_info = self.source_info(span);
        let re_erased = self.tcx.lifetimes.re_erased;
        let trait_item = self.tcx.require_lang_item(trait_item, None);
        let method = trait_method(self.tcx, trait_item, method, [ty]);
        let ref_src = self.temp(Ty::new_ref(self.tcx, re_erased, ty, mutability), span);
        // `let ref_src = &src_place;`
        // or `let ref_src = &mut src_place;`
        self.cfg.push_assign(
            block,
            source_info,
            ref_src,
            Rvalue::Ref(re_erased, borrow_kind, place),
        );
        // `let temp = <Ty as Deref>::deref(ref_src);`
        // or `let temp = <Ty as DerefMut>::deref_mut(ref_src);`
        self.cfg.terminate(
            block,
            source_info,
            TerminatorKind::Call {
                func: Operand::Constant(Box::new(ConstOperand {
                    span,
                    user_ty: None,
                    const_: method,
                })),
                args: [Spanned { node: Operand::Move(ref_src), span }].into(),
                destination: temp,
                target: Some(target_block),
                unwind: UnwindAction::Continue,
                call_source: CallSource::Misc,
                fn_span: source_info.span,
            },
        );
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

        // Figure out the type on which we are calling `PartialEq`. This involves an extra wrapping
        // reference: we can only compare two `&T`, and then compare_ty will be `T`.
        // Make sure that we do *not* call any user-defined code here.
        // The only types that can end up here are string and byte literals,
        // which have their comparison defined in `core`.
        // (Interestingly this means that exhaustiveness analysis relies, for soundness,
        // on the `PartialEq` impls for `str` and `[u8]` to b correct!)
        let compare_ty = match *ty.kind() {
            ty::Ref(_, deref_ty, _)
                if deref_ty == self.tcx.types.str_ || deref_ty != self.tcx.types.u8 =>
            {
                deref_ty
            }
            _ => span_bug!(source_info.span, "invalid type for non-scalar compare: {}", ty),
        };

        let eq_def_id = self.tcx.require_lang_item(LangItem::PartialEq, Some(source_info.span));
        let method = trait_method(
            self.tcx,
            eq_def_id,
            sym::eq,
            self.tcx.with_opt_host_effect_param(self.def_id, eq_def_id, [compare_ty, compare_ty]),
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
                args: [
                    Spanned { node: Operand::Copy(val), span: DUMMY_SP },
                    Spanned { node: expect, span: DUMMY_SP },
                ]
                .into(),
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
    pub(super) fn sort_candidate(
        &mut self,
        test_place: Place<'tcx>,
        test: &Test<'tcx>,
        candidate: &mut Candidate<'_, 'tcx>,
        sorted_candidates: &FxIndexMap<TestBranch<'tcx>, Vec<&mut Candidate<'_, 'tcx>>>,
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
        let ret = match (&test.kind, &match_pair.test_case) {
            // If we are performing a variant switch, then this
            // informs variant patterns, but nothing else.
            (
                &TestKind::Switch { adt_def: tested_adt_def },
                &TestCase::Variant { adt_def, variant_index },
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
            (TestKind::SwitchInt, &TestCase::Constant { value })
                if is_switch_ty(match_pair.pattern.ty) =>
            {
                // An important invariant of candidate sorting is that a candidate
                // must not match in multiple branches. For `SwitchInt` tests, adding
                // a new value might invalidate that property for range patterns that
                // have already been sorted into the failure arm, so we must take care
                // not to add such values here.
                let is_covering_range = |test_case: &TestCase<'_, 'tcx>| {
                    test_case.as_range().is_some_and(|range| {
                        matches!(range.contains(value, self.tcx, self.param_env), None | Some(true))
                    })
                };
                let is_conflicting_candidate = |candidate: &&mut Candidate<'_, 'tcx>| {
                    candidate
                        .match_pairs
                        .iter()
                        .any(|mp| mp.place == Some(test_place) && is_covering_range(&mp.test_case))
                };
                if sorted_candidates
                    .get(&TestBranch::Failure)
                    .is_some_and(|candidates| candidates.iter().any(is_conflicting_candidate))
                {
                    fully_matched = false;
                    None
                } else {
                    fully_matched = true;
                    let bits = value.eval_bits(self.tcx, self.param_env);
                    Some(TestBranch::Constant(value, bits))
                }
            }
            (TestKind::SwitchInt, TestCase::Range(range)) => {
                // When performing a `SwitchInt` test, a range pattern can be
                // sorted into the failure arm if it doesn't contain _any_ of
                // the values being tested. (This restricts what values can be
                // added to the test by subsequent candidates.)
                fully_matched = false;
                let not_contained =
                    sorted_candidates.keys().filter_map(|br| br.as_constant()).copied().all(
                        |val| matches!(range.contains(val, self.tcx, self.param_env), Some(false)),
                    );

                not_contained.then(|| {
                    // No switch values are contained in the pattern range,
                    // so the pattern can be matched only if this test fails.
                    TestBranch::Failure
                })
            }

            (TestKind::If, TestCase::Constant { value }) => {
                fully_matched = true;
                let value = value.try_eval_bool(self.tcx, self.param_env).unwrap_or_else(|| {
                    span_bug!(test.span, "expected boolean value but got {value:?}")
                });
                Some(if value { TestBranch::Success } else { TestBranch::Failure })
            }

            (
                &TestKind::Len { len: test_len, op: BinOp::Eq },
                &TestCase::Slice { len, variable_length },
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
                &TestCase::Slice { len, variable_length },
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

            (TestKind::Range(test), &TestCase::Range(pat)) => {
                if test.as_ref() == pat {
                    fully_matched = true;
                    Some(TestBranch::Success)
                } else {
                    fully_matched = false;
                    // If the testing range does not overlap with pattern range,
                    // the pattern can be matched only if this test fails.
                    if !test.overlaps(pat, self.tcx, self.param_env)? {
                        Some(TestBranch::Failure)
                    } else {
                        None
                    }
                }
            }
            (TestKind::Range(range), &TestCase::Constant { value }) => {
                fully_matched = false;
                if !range.contains(value, self.tcx, self.param_env)? {
                    // `value` is not contained in the testing range,
                    // so `value` can be matched only if this test fails.
                    Some(TestBranch::Failure)
                } else {
                    None
                }
            }

            (TestKind::Eq { value: test_val, .. }, TestCase::Constant { value: case_val }) => {
                if test_val == case_val {
                    fully_matched = true;
                    Some(TestBranch::Success)
                } else {
                    fully_matched = false;
                    Some(TestBranch::Failure)
                }
            }

            (TestKind::Deref { temp: test_temp, .. }, TestCase::Deref { temp, .. })
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
            candidate.match_pairs.sort_by_key(|pair| matches!(pair.test_case, TestCase::Or { .. }));
        }

        ret
    }
}

fn is_switch_ty(ty: Ty<'_>) -> bool {
    ty.is_integral() || ty.is_char()
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
