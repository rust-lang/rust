// Testing candidates
//
// After candidates have been simplified, the only match pairs that
// remain are those that require some sort of test. The functions here
// identify what tests are needed, perform the tests, and then filter
// the candidates based on the result.

use std::sync::Arc;

use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::{LangItem, RangeEnd};
use rustc_middle::mir::*;
use rustc_middle::ty::util::IntTypeExt;
use rustc_middle::ty::{self, GenericArg, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_span::def_id::DefId;
use rustc_span::source_map::Spanned;
use rustc_span::{DUMMY_SP, Span, Symbol, sym};
use tracing::{debug, instrument};

use crate::builder::Builder;
use crate::builder::matches::{
    MatchPairTree, PatConstKind, SliceLenOp, Test, TestBranch, TestKind, TestableCase,
};

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Identifies what test is needed to decide if `match_pair` is applicable.
    ///
    /// It is a bug to call this with a not-fully-simplified pattern.
    pub(super) fn pick_test_for_match_pair(
        &mut self,
        match_pair: &MatchPairTree<'tcx>,
    ) -> Test<'tcx> {
        let kind = match match_pair.testable_case {
            TestableCase::Variant { adt_def, variant_index: _ } => TestKind::Switch { adt_def },

            TestableCase::Constant { value: _, kind: PatConstKind::Bool } => TestKind::If,
            TestableCase::Constant { value: _, kind: PatConstKind::IntOrChar } => {
                TestKind::SwitchInt
            }
            TestableCase::Constant { value, kind: PatConstKind::Float } => {
                TestKind::Eq { value, cast_ty: match_pair.pattern_ty }
            }
            TestableCase::Constant { value, kind: PatConstKind::Other } => {
                TestKind::Eq { value, cast_ty: match_pair.pattern_ty }
            }

            TestableCase::Range(ref range) => {
                assert_eq!(range.ty, match_pair.pattern_ty);
                TestKind::Range(Arc::clone(range))
            }

            TestableCase::Slice { len, op } => TestKind::SliceLen { len, op },

            TestableCase::Deref { temp, mutability } => TestKind::Deref { temp, mutability },

            TestableCase::Never => TestKind::Never,

            // Or-patterns are not tested directly; instead they are expanded into subcandidates,
            // which are then distinguished by testing whatever non-or patterns they contain.
            TestableCase::Or { .. } => bug!("or-patterns should have already been handled"),
        };

        Test { span: match_pair.pattern_span, kind }
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
                        if let TestBranch::Constant(value) = branch {
                            let bits = value.to_leaf().to_bits_unchecked();
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

            TestKind::Eq { value, mut cast_ty } => {
                let tcx = self.tcx;
                let success_block = target_block(TestBranch::Success);
                let fail_block = target_block(TestBranch::Failure);

                let mut expect_ty = value.ty;
                let mut expect = self.literal_operand(test.span, Const::from_ty_value(tcx, value));

                let mut place = place;
                let mut block = block;
                match cast_ty.kind() {
                    ty::Str => {
                        // String literal patterns may have type `str` if `deref_patterns` is
                        // enabled, in order to allow `deref!("..."): String`. In this case, `value`
                        // is of type `&str`, so we compare it to `&place`.
                        if !tcx.features().deref_patterns() {
                            span_bug!(
                                test.span,
                                "matching on `str` went through without enabling deref_patterns"
                            );
                        }
                        let re_erased = tcx.lifetimes.re_erased;
                        let ref_str_ty = Ty::new_imm_ref(tcx, re_erased, tcx.types.str_);
                        let ref_place = self.temp(ref_str_ty, test.span);
                        // `let ref_place: &str = &place;`
                        self.cfg.push_assign(
                            block,
                            self.source_info(test.span),
                            ref_place,
                            Rvalue::Ref(re_erased, BorrowKind::Shared, place),
                        );
                        place = ref_place;
                        cast_ty = ref_str_ty;
                    }
                    ty::Adt(def, _) if tcx.is_lang_item(def.did(), LangItem::String) => {
                        if !tcx.features().string_deref_patterns() {
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
                            cast_ty,
                            ref_str,
                            test.span,
                        );
                        // Since we generated a `ref_str = <String as Deref>::deref(&place) -> eq_block` terminator,
                        // we need to add all further statements to `eq_block`.
                        // Similarly, the normal test code should be generated for the `&str`, instead of the `String`.
                        block = eq_block;
                        place = ref_str;
                        cast_ty = ref_str_ty;
                    }
                    &ty::Pat(base, _) => {
                        assert_eq!(cast_ty, value.ty);
                        assert!(base.is_trivially_pure_clone_copy());

                        let transmuted_place = self.temp(base, test.span);
                        self.cfg.push_assign(
                            block,
                            self.source_info(scrutinee_span),
                            transmuted_place,
                            Rvalue::Cast(CastKind::Transmute, Operand::Copy(place), base),
                        );

                        let transmuted_expect = self.temp(base, test.span);
                        self.cfg.push_assign(
                            block,
                            self.source_info(test.span),
                            transmuted_expect,
                            Rvalue::Cast(CastKind::Transmute, expect, base),
                        );

                        place = transmuted_place;
                        expect = Operand::Copy(transmuted_expect);
                        cast_ty = base;
                        expect_ty = base;
                    }
                    _ => {}
                }

                assert_eq!(expect_ty, cast_ty);
                if !cast_ty.is_scalar() {
                    // Use `PartialEq::eq` instead of `BinOp::Eq`
                    // (the binop can only handle primitives)
                    // Make sure that we do *not* call any user-defined code here.
                    // The only type that can end up here is string literals, which have their
                    // comparison defined in `core`.
                    // (Interestingly this means that exhaustiveness analysis relies, for soundness,
                    // on the `PartialEq` impl for `str` to b correct!)
                    match *cast_ty.kind() {
                        ty::Ref(_, deref_ty, _) if deref_ty == self.tcx.types.str_ => {}
                        _ => {
                            span_bug!(
                                source_info.span,
                                "invalid type for non-scalar compare: {cast_ty}"
                            )
                        }
                    };
                    self.string_compare(
                        block,
                        success_block,
                        fail_block,
                        source_info,
                        expect,
                        Operand::Copy(place),
                    );
                } else {
                    self.compare(
                        block,
                        success_block,
                        fail_block,
                        source_info,
                        BinOp::Eq,
                        expect,
                        Operand::Copy(place),
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
                    let lo = ty::Value { ty: range.ty, valtree: lo };
                    let lo = self.literal_operand(test.span, Const::from_ty_value(self.tcx, lo));
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
                    let hi = ty::Value { ty: range.ty, valtree: hi };
                    let hi = self.literal_operand(test.span, Const::from_ty_value(self.tcx, hi));
                    let op = match range.end {
                        RangeEnd::Included => BinOp::Le,
                        RangeEnd::Excluded => BinOp::Lt,
                    };
                    self.compare(intermediate_block, success, fail, source_info, op, val, hi);
                }
            }

            TestKind::SliceLen { len, op } => {
                let usize_ty = self.tcx.types.usize;
                let actual = self.temp(usize_ty, test.span);

                // actual = len(place)
                let length_op = self.len_of_slice_or_array(block, place, test.span, source_info);
                self.cfg.push_assign(block, source_info, actual, Rvalue::Use(length_op));

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
                    match op {
                        SliceLenOp::Equal => BinOp::Eq,
                        SliceLenOp::GreaterOrEqual => BinOp::Ge,
                    },
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
        let trait_item = self.tcx.require_lang_item(trait_item, span);
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

    /// Compare two values of type `&str` using `<str as std::cmp::PartialEq>::eq`.
    fn string_compare(
        &mut self,
        block: BasicBlock,
        success_block: BasicBlock,
        fail_block: BasicBlock,
        source_info: SourceInfo,
        expect: Operand<'tcx>,
        val: Operand<'tcx>,
    ) {
        let str_ty = self.tcx.types.str_;
        let eq_def_id = self.tcx.require_lang_item(LangItem::PartialEq, source_info.span);
        let method = trait_method(self.tcx, eq_def_id, sym::eq, [str_ty, str_ty]);

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
                    Spanned { node: val, span: DUMMY_SP },
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
        .find(|item| item.is_fn())
        .expect("trait method not found");

    let method_ty = Ty::new_fn_def(tcx, item.def_id, args);

    Const::zero_sized(method_ty)
}
