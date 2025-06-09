//! Conversion of internal Rust compiler `mir` items to stable ones.

use rustc_middle::mir::mono::MonoItem;
use rustc_middle::{bug, mir};
use rustc_smir::Tables;
use rustc_smir::bridge::SmirError;
use rustc_smir::context::SmirCtxt;
use stable_mir::compiler_interface::BridgeTys;
use stable_mir::mir::alloc::GlobalAlloc;
use stable_mir::mir::{ConstOperand, Statement, UserTypeProjection, VarDebugInfoFragment};
use stable_mir::ty::{Allocation, ConstantKind, MirConst};
use stable_mir::unstable::Stable;
use stable_mir::{Error, alloc, opaque};

use crate::{rustc_smir, stable_mir};

impl<'tcx> Stable<'tcx> for mir::Body<'tcx> {
    type T = stable_mir::mir::Body;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        stable_mir::mir::Body::new(
            self.basic_blocks
                .iter()
                .map(|block| stable_mir::mir::BasicBlock {
                    terminator: block.terminator().stable(tables, cx),
                    statements: block
                        .statements
                        .iter()
                        .map(|statement| statement.stable(tables, cx))
                        .collect(),
                })
                .collect(),
            self.local_decls
                .iter()
                .map(|decl| stable_mir::mir::LocalDecl {
                    ty: decl.ty.stable(tables, cx),
                    span: decl.source_info.span.stable(tables, cx),
                    mutability: decl.mutability.stable(tables, cx),
                })
                .collect(),
            self.arg_count,
            self.var_debug_info.iter().map(|info| info.stable(tables, cx)).collect(),
            self.spread_arg.stable(tables, cx),
            self.span.stable(tables, cx),
        )
    }
}

impl<'tcx> Stable<'tcx> for mir::VarDebugInfo<'tcx> {
    type T = stable_mir::mir::VarDebugInfo;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        stable_mir::mir::VarDebugInfo {
            name: self.name.to_string(),
            source_info: self.source_info.stable(tables, cx),
            composite: self.composite.as_ref().map(|composite| composite.stable(tables, cx)),
            value: self.value.stable(tables, cx),
            argument_index: self.argument_index,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::Statement<'tcx> {
    type T = stable_mir::mir::Statement;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        Statement {
            kind: self.kind.stable(tables, cx),
            span: self.source_info.span.stable(tables, cx),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::SourceInfo {
    type T = stable_mir::mir::SourceInfo;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        stable_mir::mir::SourceInfo { span: self.span.stable(tables, cx), scope: self.scope.into() }
    }
}

impl<'tcx> Stable<'tcx> for mir::VarDebugInfoFragment<'tcx> {
    type T = stable_mir::mir::VarDebugInfoFragment;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        VarDebugInfoFragment {
            ty: self.ty.stable(tables, cx),
            projection: self.projection.iter().map(|e| e.stable(tables, cx)).collect(),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::VarDebugInfoContents<'tcx> {
    type T = stable_mir::mir::VarDebugInfoContents;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match self {
            mir::VarDebugInfoContents::Place(place) => {
                stable_mir::mir::VarDebugInfoContents::Place(place.stable(tables, cx))
            }
            mir::VarDebugInfoContents::Const(const_operand) => {
                let op = ConstOperand {
                    span: const_operand.span.stable(tables, cx),
                    user_ty: const_operand.user_ty.map(|index| index.as_usize()),
                    const_: const_operand.const_.stable(tables, cx),
                };
                stable_mir::mir::VarDebugInfoContents::Const(op)
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::StatementKind<'tcx> {
    type T = stable_mir::mir::StatementKind;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match self {
            mir::StatementKind::Assign(assign) => stable_mir::mir::StatementKind::Assign(
                assign.0.stable(tables, cx),
                assign.1.stable(tables, cx),
            ),
            mir::StatementKind::FakeRead(fake_read_place) => {
                stable_mir::mir::StatementKind::FakeRead(
                    fake_read_place.0.stable(tables, cx),
                    fake_read_place.1.stable(tables, cx),
                )
            }
            mir::StatementKind::SetDiscriminant { place, variant_index } => {
                stable_mir::mir::StatementKind::SetDiscriminant {
                    place: place.as_ref().stable(tables, cx),
                    variant_index: variant_index.stable(tables, cx),
                }
            }
            mir::StatementKind::Deinit(place) => {
                stable_mir::mir::StatementKind::Deinit(place.stable(tables, cx))
            }

            mir::StatementKind::StorageLive(place) => {
                stable_mir::mir::StatementKind::StorageLive(place.stable(tables, cx))
            }

            mir::StatementKind::StorageDead(place) => {
                stable_mir::mir::StatementKind::StorageDead(place.stable(tables, cx))
            }
            mir::StatementKind::Retag(retag, place) => stable_mir::mir::StatementKind::Retag(
                retag.stable(tables, cx),
                place.stable(tables, cx),
            ),
            mir::StatementKind::PlaceMention(place) => {
                stable_mir::mir::StatementKind::PlaceMention(place.stable(tables, cx))
            }
            mir::StatementKind::AscribeUserType(place_projection, variance) => {
                stable_mir::mir::StatementKind::AscribeUserType {
                    place: place_projection.as_ref().0.stable(tables, cx),
                    projections: place_projection.as_ref().1.stable(tables, cx),
                    variance: variance.stable(tables, cx),
                }
            }
            mir::StatementKind::Coverage(coverage) => {
                stable_mir::mir::StatementKind::Coverage(opaque(coverage))
            }
            mir::StatementKind::Intrinsic(intrinstic) => {
                stable_mir::mir::StatementKind::Intrinsic(intrinstic.stable(tables, cx))
            }
            mir::StatementKind::ConstEvalCounter => {
                stable_mir::mir::StatementKind::ConstEvalCounter
            }
            // BackwardIncompatibleDropHint has no semantics, so it is translated to Nop.
            mir::StatementKind::BackwardIncompatibleDropHint { .. } => {
                stable_mir::mir::StatementKind::Nop
            }
            mir::StatementKind::Nop => stable_mir::mir::StatementKind::Nop,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::Rvalue<'tcx> {
    type T = stable_mir::mir::Rvalue;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use rustc_middle::mir::Rvalue::*;
        match self {
            Use(op) => stable_mir::mir::Rvalue::Use(op.stable(tables, cx)),
            Repeat(op, len) => {
                let len = len.stable(tables, cx);
                stable_mir::mir::Rvalue::Repeat(op.stable(tables, cx), len)
            }
            Ref(region, kind, place) => stable_mir::mir::Rvalue::Ref(
                region.stable(tables, cx),
                kind.stable(tables, cx),
                place.stable(tables, cx),
            ),
            ThreadLocalRef(def_id) => {
                stable_mir::mir::Rvalue::ThreadLocalRef(tables.crate_item(*def_id))
            }
            RawPtr(mutability, place) => stable_mir::mir::Rvalue::AddressOf(
                mutability.stable(tables, cx),
                place.stable(tables, cx),
            ),
            Len(place) => stable_mir::mir::Rvalue::Len(place.stable(tables, cx)),
            Cast(cast_kind, op, ty) => stable_mir::mir::Rvalue::Cast(
                cast_kind.stable(tables, cx),
                op.stable(tables, cx),
                ty.stable(tables, cx),
            ),
            BinaryOp(bin_op, ops) => {
                if let Some(bin_op) = bin_op.overflowing_to_wrapping() {
                    stable_mir::mir::Rvalue::CheckedBinaryOp(
                        bin_op.stable(tables, cx),
                        ops.0.stable(tables, cx),
                        ops.1.stable(tables, cx),
                    )
                } else {
                    stable_mir::mir::Rvalue::BinaryOp(
                        bin_op.stable(tables, cx),
                        ops.0.stable(tables, cx),
                        ops.1.stable(tables, cx),
                    )
                }
            }
            NullaryOp(null_op, ty) => stable_mir::mir::Rvalue::NullaryOp(
                null_op.stable(tables, cx),
                ty.stable(tables, cx),
            ),
            UnaryOp(un_op, op) => {
                stable_mir::mir::Rvalue::UnaryOp(un_op.stable(tables, cx), op.stable(tables, cx))
            }
            Discriminant(place) => stable_mir::mir::Rvalue::Discriminant(place.stable(tables, cx)),
            Aggregate(agg_kind, operands) => {
                let operands = operands.iter().map(|op| op.stable(tables, cx)).collect();
                stable_mir::mir::Rvalue::Aggregate(agg_kind.stable(tables, cx), operands)
            }
            ShallowInitBox(op, ty) => stable_mir::mir::Rvalue::ShallowInitBox(
                op.stable(tables, cx),
                ty.stable(tables, cx),
            ),
            CopyForDeref(place) => stable_mir::mir::Rvalue::CopyForDeref(place.stable(tables, cx)),
            WrapUnsafeBinder(..) => todo!("FIXME(unsafe_binders):"),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::Mutability {
    type T = stable_mir::mir::Mutability;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &SmirCtxt<'_, BridgeTys>) -> Self::T {
        use rustc_hir::Mutability::*;
        match *self {
            Not => stable_mir::mir::Mutability::Not,
            Mut => stable_mir::mir::Mutability::Mut,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::RawPtrKind {
    type T = stable_mir::mir::RawPtrKind;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &SmirCtxt<'_, BridgeTys>) -> Self::T {
        use mir::RawPtrKind::*;
        match *self {
            Const => stable_mir::mir::RawPtrKind::Const,
            Mut => stable_mir::mir::RawPtrKind::Mut,
            FakeForPtrMetadata => stable_mir::mir::RawPtrKind::FakeForPtrMetadata,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::BorrowKind {
    type T = stable_mir::mir::BorrowKind;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use rustc_middle::mir::BorrowKind::*;
        match *self {
            Shared => stable_mir::mir::BorrowKind::Shared,
            Fake(kind) => stable_mir::mir::BorrowKind::Fake(kind.stable(tables, cx)),
            Mut { kind } => stable_mir::mir::BorrowKind::Mut { kind: kind.stable(tables, cx) },
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::MutBorrowKind {
    type T = stable_mir::mir::MutBorrowKind;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &SmirCtxt<'_, BridgeTys>) -> Self::T {
        use rustc_middle::mir::MutBorrowKind::*;
        match *self {
            Default => stable_mir::mir::MutBorrowKind::Default,
            TwoPhaseBorrow => stable_mir::mir::MutBorrowKind::TwoPhaseBorrow,
            ClosureCapture => stable_mir::mir::MutBorrowKind::ClosureCapture,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::FakeBorrowKind {
    type T = stable_mir::mir::FakeBorrowKind;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &SmirCtxt<'_, BridgeTys>) -> Self::T {
        use rustc_middle::mir::FakeBorrowKind::*;
        match *self {
            Deep => stable_mir::mir::FakeBorrowKind::Deep,
            Shallow => stable_mir::mir::FakeBorrowKind::Shallow,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::NullOp<'tcx> {
    type T = stable_mir::mir::NullOp;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use rustc_middle::mir::NullOp::*;
        match self {
            SizeOf => stable_mir::mir::NullOp::SizeOf,
            AlignOf => stable_mir::mir::NullOp::AlignOf,
            OffsetOf(indices) => stable_mir::mir::NullOp::OffsetOf(
                indices.iter().map(|idx| idx.stable(tables, cx)).collect(),
            ),
            UbChecks => stable_mir::mir::NullOp::UbChecks,
            ContractChecks => stable_mir::mir::NullOp::ContractChecks,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::CastKind {
    type T = stable_mir::mir::CastKind;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use rustc_middle::mir::CastKind::*;
        match self {
            PointerExposeProvenance => stable_mir::mir::CastKind::PointerExposeAddress,
            PointerWithExposedProvenance => stable_mir::mir::CastKind::PointerWithExposedProvenance,
            PointerCoercion(c, _) => stable_mir::mir::CastKind::PointerCoercion(c.stable(tables, cx)),
            IntToInt => stable_mir::mir::CastKind::IntToInt,
            FloatToInt => stable_mir::mir::CastKind::FloatToInt,
            FloatToFloat => stable_mir::mir::CastKind::FloatToFloat,
            IntToFloat => stable_mir::mir::CastKind::IntToFloat,
            PtrToPtr => stable_mir::mir::CastKind::PtrToPtr,
            FnPtrToPtr => stable_mir::mir::CastKind::FnPtrToPtr,
            Transmute => stable_mir::mir::CastKind::Transmute,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::FakeReadCause {
    type T = stable_mir::mir::FakeReadCause;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &SmirCtxt<'_, BridgeTys>) -> Self::T {
        use rustc_middle::mir::FakeReadCause::*;
        match self {
            ForMatchGuard => stable_mir::mir::FakeReadCause::ForMatchGuard,
            ForMatchedPlace(local_def_id) => {
                stable_mir::mir::FakeReadCause::ForMatchedPlace(opaque(local_def_id))
            }
            ForGuardBinding => stable_mir::mir::FakeReadCause::ForGuardBinding,
            ForLet(local_def_id) => stable_mir::mir::FakeReadCause::ForLet(opaque(local_def_id)),
            ForIndex => stable_mir::mir::FakeReadCause::ForIndex,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::Operand<'tcx> {
    type T = stable_mir::mir::Operand;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use rustc_middle::mir::Operand::*;
        match self {
            Copy(place) => stable_mir::mir::Operand::Copy(place.stable(tables, cx)),
            Move(place) => stable_mir::mir::Operand::Move(place.stable(tables, cx)),
            Constant(c) => stable_mir::mir::Operand::Constant(c.stable(tables, cx)),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::ConstOperand<'tcx> {
    type T = stable_mir::mir::ConstOperand;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        stable_mir::mir::ConstOperand {
            span: self.span.stable(tables, cx),
            user_ty: self.user_ty.map(|u| u.as_usize()).or(None),
            const_: self.const_.stable(tables, cx),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::Place<'tcx> {
    type T = stable_mir::mir::Place;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        stable_mir::mir::Place {
            local: self.local.as_usize(),
            projection: self.projection.iter().map(|e| e.stable(tables, cx)).collect(),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::PlaceElem<'tcx> {
    type T = stable_mir::mir::ProjectionElem;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use rustc_middle::mir::ProjectionElem::*;
        match self {
            Deref => stable_mir::mir::ProjectionElem::Deref,
            Field(idx, ty) => stable_mir::mir::ProjectionElem::Field(
                idx.stable(tables, cx),
                ty.stable(tables, cx),
            ),
            Index(local) => stable_mir::mir::ProjectionElem::Index(local.stable(tables, cx)),
            ConstantIndex { offset, min_length, from_end } => {
                stable_mir::mir::ProjectionElem::ConstantIndex {
                    offset: *offset,
                    min_length: *min_length,
                    from_end: *from_end,
                }
            }
            Subslice { from, to, from_end } => stable_mir::mir::ProjectionElem::Subslice {
                from: *from,
                to: *to,
                from_end: *from_end,
            },
            // MIR includes an `Option<Symbol>` argument for `Downcast` that is the name of the
            // variant, used for printing MIR. However this information should also be accessible
            // via a lookup using the `VariantIdx`. The `Option<Symbol>` argument is therefore
            // dropped when converting to Stable MIR. A brief justification for this decision can be
            // found at https://github.com/rust-lang/rust/pull/117517#issuecomment-1811683486
            Downcast(_, idx) => stable_mir::mir::ProjectionElem::Downcast(idx.stable(tables, cx)),
            OpaqueCast(ty) => stable_mir::mir::ProjectionElem::OpaqueCast(ty.stable(tables, cx)),
            Subtype(ty) => stable_mir::mir::ProjectionElem::Subtype(ty.stable(tables, cx)),
            UnwrapUnsafeBinder(..) => todo!("FIXME(unsafe_binders):"),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::UserTypeProjection {
    type T = stable_mir::mir::UserTypeProjection;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &SmirCtxt<'_, BridgeTys>) -> Self::T {
        UserTypeProjection { base: self.base.as_usize(), projection: opaque(&self.projs) }
    }
}

impl<'tcx> Stable<'tcx> for mir::Local {
    type T = stable_mir::mir::Local;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &SmirCtxt<'_, BridgeTys>) -> Self::T {
        self.as_usize()
    }
}

impl<'tcx> Stable<'tcx> for mir::RetagKind {
    type T = stable_mir::mir::RetagKind;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &SmirCtxt<'_, BridgeTys>) -> Self::T {
        use rustc_middle::mir::RetagKind;
        match self {
            RetagKind::FnEntry => stable_mir::mir::RetagKind::FnEntry,
            RetagKind::TwoPhase => stable_mir::mir::RetagKind::TwoPhase,
            RetagKind::Raw => stable_mir::mir::RetagKind::Raw,
            RetagKind::Default => stable_mir::mir::RetagKind::Default,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::UnwindAction {
    type T = stable_mir::mir::UnwindAction;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &SmirCtxt<'_, BridgeTys>) -> Self::T {
        use rustc_middle::mir::UnwindAction;
        match self {
            UnwindAction::Continue => stable_mir::mir::UnwindAction::Continue,
            UnwindAction::Unreachable => stable_mir::mir::UnwindAction::Unreachable,
            UnwindAction::Terminate(_) => stable_mir::mir::UnwindAction::Terminate,
            UnwindAction::Cleanup(bb) => stable_mir::mir::UnwindAction::Cleanup(bb.as_usize()),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::NonDivergingIntrinsic<'tcx> {
    type T = stable_mir::mir::NonDivergingIntrinsic;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use rustc_middle::mir::NonDivergingIntrinsic;
        use stable_mir::mir::CopyNonOverlapping;
        match self {
            NonDivergingIntrinsic::Assume(op) => {
                stable_mir::mir::NonDivergingIntrinsic::Assume(op.stable(tables, cx))
            }
            NonDivergingIntrinsic::CopyNonOverlapping(copy_non_overlapping) => {
                stable_mir::mir::NonDivergingIntrinsic::CopyNonOverlapping(CopyNonOverlapping {
                    src: copy_non_overlapping.src.stable(tables, cx),
                    dst: copy_non_overlapping.dst.stable(tables, cx),
                    count: copy_non_overlapping.count.stable(tables, cx),
                })
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::AssertMessage<'tcx> {
    type T = stable_mir::mir::AssertMessage;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use rustc_middle::mir::AssertKind;
        match self {
            AssertKind::BoundsCheck { len, index } => stable_mir::mir::AssertMessage::BoundsCheck {
                len: len.stable(tables, cx),
                index: index.stable(tables, cx),
            },
            AssertKind::Overflow(bin_op, op1, op2) => stable_mir::mir::AssertMessage::Overflow(
                bin_op.stable(tables, cx),
                op1.stable(tables, cx),
                op2.stable(tables, cx),
            ),
            AssertKind::OverflowNeg(op) => {
                stable_mir::mir::AssertMessage::OverflowNeg(op.stable(tables, cx))
            }
            AssertKind::DivisionByZero(op) => {
                stable_mir::mir::AssertMessage::DivisionByZero(op.stable(tables, cx))
            }
            AssertKind::RemainderByZero(op) => {
                stable_mir::mir::AssertMessage::RemainderByZero(op.stable(tables, cx))
            }
            AssertKind::ResumedAfterReturn(coroutine) => {
                stable_mir::mir::AssertMessage::ResumedAfterReturn(coroutine.stable(tables, cx))
            }
            AssertKind::ResumedAfterPanic(coroutine) => {
                stable_mir::mir::AssertMessage::ResumedAfterPanic(coroutine.stable(tables, cx))
            }
            AssertKind::ResumedAfterDrop(coroutine) => {
                stable_mir::mir::AssertMessage::ResumedAfterDrop(coroutine.stable(tables, cx))
            }
            AssertKind::MisalignedPointerDereference { required, found } => {
                stable_mir::mir::AssertMessage::MisalignedPointerDereference {
                    required: required.stable(tables, cx),
                    found: found.stable(tables, cx),
                }
            }
            AssertKind::NullPointerDereference => {
                stable_mir::mir::AssertMessage::NullPointerDereference
            }
            AssertKind::InvalidEnumConstruction(source) => {
                stable_mir::mir::AssertMessage::InvalidEnumConstruction(source.stable(tables, cx))
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::BinOp {
    type T = stable_mir::mir::BinOp;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &SmirCtxt<'_, BridgeTys>) -> Self::T {
        use rustc_middle::mir::BinOp;
        match self {
            BinOp::Add => stable_mir::mir::BinOp::Add,
            BinOp::AddUnchecked => stable_mir::mir::BinOp::AddUnchecked,
            BinOp::AddWithOverflow => bug!("AddWithOverflow should have been translated already"),
            BinOp::Sub => stable_mir::mir::BinOp::Sub,
            BinOp::SubUnchecked => stable_mir::mir::BinOp::SubUnchecked,
            BinOp::SubWithOverflow => bug!("AddWithOverflow should have been translated already"),
            BinOp::Mul => stable_mir::mir::BinOp::Mul,
            BinOp::MulUnchecked => stable_mir::mir::BinOp::MulUnchecked,
            BinOp::MulWithOverflow => bug!("AddWithOverflow should have been translated already"),
            BinOp::Div => stable_mir::mir::BinOp::Div,
            BinOp::Rem => stable_mir::mir::BinOp::Rem,
            BinOp::BitXor => stable_mir::mir::BinOp::BitXor,
            BinOp::BitAnd => stable_mir::mir::BinOp::BitAnd,
            BinOp::BitOr => stable_mir::mir::BinOp::BitOr,
            BinOp::Shl => stable_mir::mir::BinOp::Shl,
            BinOp::ShlUnchecked => stable_mir::mir::BinOp::ShlUnchecked,
            BinOp::Shr => stable_mir::mir::BinOp::Shr,
            BinOp::ShrUnchecked => stable_mir::mir::BinOp::ShrUnchecked,
            BinOp::Eq => stable_mir::mir::BinOp::Eq,
            BinOp::Lt => stable_mir::mir::BinOp::Lt,
            BinOp::Le => stable_mir::mir::BinOp::Le,
            BinOp::Ne => stable_mir::mir::BinOp::Ne,
            BinOp::Ge => stable_mir::mir::BinOp::Ge,
            BinOp::Gt => stable_mir::mir::BinOp::Gt,
            BinOp::Cmp => stable_mir::mir::BinOp::Cmp,
            BinOp::Offset => stable_mir::mir::BinOp::Offset,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::UnOp {
    type T = stable_mir::mir::UnOp;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &SmirCtxt<'_, BridgeTys>) -> Self::T {
        use rustc_middle::mir::UnOp;
        match self {
            UnOp::Not => stable_mir::mir::UnOp::Not,
            UnOp::Neg => stable_mir::mir::UnOp::Neg,
            UnOp::PtrMetadata => stable_mir::mir::UnOp::PtrMetadata,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::AggregateKind<'tcx> {
    type T = stable_mir::mir::AggregateKind;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match self {
            mir::AggregateKind::Array(ty) => {
                stable_mir::mir::AggregateKind::Array(ty.stable(tables, cx))
            }
            mir::AggregateKind::Tuple => stable_mir::mir::AggregateKind::Tuple,
            mir::AggregateKind::Adt(def_id, var_idx, generic_arg, user_ty_index, field_idx) => {
                stable_mir::mir::AggregateKind::Adt(
                    tables.adt_def(*def_id),
                    var_idx.stable(tables, cx),
                    generic_arg.stable(tables, cx),
                    user_ty_index.map(|idx| idx.index()),
                    field_idx.map(|idx| idx.index()),
                )
            }
            mir::AggregateKind::Closure(def_id, generic_arg) => {
                stable_mir::mir::AggregateKind::Closure(
                    tables.closure_def(*def_id),
                    generic_arg.stable(tables, cx),
                )
            }
            mir::AggregateKind::Coroutine(def_id, generic_arg) => {
                stable_mir::mir::AggregateKind::Coroutine(
                    tables.coroutine_def(*def_id),
                    generic_arg.stable(tables, cx),
                    cx.coroutine_movability(*def_id).stable(tables, cx),
                )
            }
            mir::AggregateKind::CoroutineClosure(def_id, generic_args) => {
                stable_mir::mir::AggregateKind::CoroutineClosure(
                    tables.coroutine_closure_def(*def_id),
                    generic_args.stable(tables, cx),
                )
            }
            mir::AggregateKind::RawPtr(ty, mutability) => stable_mir::mir::AggregateKind::RawPtr(
                ty.stable(tables, cx),
                mutability.stable(tables, cx),
            ),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::InlineAsmOperand<'tcx> {
    type T = stable_mir::mir::InlineAsmOperand;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use rustc_middle::mir::InlineAsmOperand;

        let (in_value, out_place) = match self {
            InlineAsmOperand::In { value, .. } => (Some(value.stable(tables, cx)), None),
            InlineAsmOperand::Out { place, .. } => {
                (None, place.map(|place| place.stable(tables, cx)))
            }
            InlineAsmOperand::InOut { in_value, out_place, .. } => {
                (Some(in_value.stable(tables, cx)), out_place.map(|place| place.stable(tables, cx)))
            }
            InlineAsmOperand::Const { .. }
            | InlineAsmOperand::SymFn { .. }
            | InlineAsmOperand::SymStatic { .. }
            | InlineAsmOperand::Label { .. } => (None, None),
        };

        stable_mir::mir::InlineAsmOperand { in_value, out_place, raw_rpr: format!("{self:?}") }
    }
}

impl<'tcx> Stable<'tcx> for mir::Terminator<'tcx> {
    type T = stable_mir::mir::Terminator;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use stable_mir::mir::Terminator;
        Terminator {
            kind: self.kind.stable(tables, cx),
            span: self.source_info.span.stable(tables, cx),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::TerminatorKind<'tcx> {
    type T = stable_mir::mir::TerminatorKind;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use stable_mir::mir::TerminatorKind;
        match self {
            mir::TerminatorKind::Goto { target } => {
                TerminatorKind::Goto { target: target.as_usize() }
            }
            mir::TerminatorKind::SwitchInt { discr, targets } => TerminatorKind::SwitchInt {
                discr: discr.stable(tables, cx),
                targets: {
                    let branches = targets.iter().map(|(val, target)| (val, target.as_usize()));
                    stable_mir::mir::SwitchTargets::new(
                        branches.collect(),
                        targets.otherwise().as_usize(),
                    )
                },
            },
            mir::TerminatorKind::UnwindResume => TerminatorKind::Resume,
            mir::TerminatorKind::UnwindTerminate(_) => TerminatorKind::Abort,
            mir::TerminatorKind::Return => TerminatorKind::Return,
            mir::TerminatorKind::Unreachable => TerminatorKind::Unreachable,
            mir::TerminatorKind::Drop {
                place,
                target,
                unwind,
                replace: _,
                drop: _,
                async_fut: _,
            } => TerminatorKind::Drop {
                place: place.stable(tables, cx),
                target: target.as_usize(),
                unwind: unwind.stable(tables, cx),
            },
            mir::TerminatorKind::Call {
                func,
                args,
                destination,
                target,
                unwind,
                call_source: _,
                fn_span: _,
            } => TerminatorKind::Call {
                func: func.stable(tables, cx),
                args: args.iter().map(|arg| arg.node.stable(tables, cx)).collect(),
                destination: destination.stable(tables, cx),
                target: target.map(|t| t.as_usize()),
                unwind: unwind.stable(tables, cx),
            },
            mir::TerminatorKind::TailCall { func: _, args: _, fn_span: _ } => todo!(),
            mir::TerminatorKind::Assert { cond, expected, msg, target, unwind } => {
                TerminatorKind::Assert {
                    cond: cond.stable(tables, cx),
                    expected: *expected,
                    msg: msg.stable(tables, cx),
                    target: target.as_usize(),
                    unwind: unwind.stable(tables, cx),
                }
            }
            mir::TerminatorKind::InlineAsm {
                asm_macro: _,
                template,
                operands,
                options,
                line_spans,
                targets,
                unwind,
            } => TerminatorKind::InlineAsm {
                template: format!("{template:?}"),
                operands: operands.iter().map(|operand| operand.stable(tables, cx)).collect(),
                options: format!("{options:?}"),
                line_spans: format!("{line_spans:?}"),
                // FIXME: Figure out how to do labels in SMIR
                destination: targets.first().map(|d| d.as_usize()),
                unwind: unwind.stable(tables, cx),
            },
            mir::TerminatorKind::Yield { .. }
            | mir::TerminatorKind::CoroutineDrop
            | mir::TerminatorKind::FalseEdge { .. }
            | mir::TerminatorKind::FalseUnwind { .. } => unreachable!(),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::interpret::ConstAllocation<'tcx> {
    type T = Allocation;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        self.inner().stable(tables, cx)
    }
}

impl<'tcx> Stable<'tcx> for mir::interpret::Allocation {
    type T = stable_mir::ty::Allocation;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use rustc_smir::context::SmirAllocRange;
        alloc::allocation_filter(
            self,
            cx.alloc_range(rustc_abi::Size::ZERO, self.size()),
            tables,
            cx,
        )
    }
}

impl<'tcx> Stable<'tcx> for mir::interpret::AllocId {
    type T = stable_mir::mir::alloc::AllocId;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        _: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        tables.create_alloc_id(*self)
    }
}

impl<'tcx> Stable<'tcx> for mir::interpret::GlobalAlloc<'tcx> {
    type T = GlobalAlloc;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match self {
            mir::interpret::GlobalAlloc::Function { instance, .. } => {
                GlobalAlloc::Function(instance.stable(tables, cx))
            }
            mir::interpret::GlobalAlloc::VTable(ty, dyn_ty) => {
                // FIXME: Should we record the whole vtable?
                GlobalAlloc::VTable(ty.stable(tables, cx), dyn_ty.principal().stable(tables, cx))
            }
            mir::interpret::GlobalAlloc::Static(def) => {
                GlobalAlloc::Static(tables.static_def(*def))
            }
            mir::interpret::GlobalAlloc::Memory(alloc) => {
                GlobalAlloc::Memory(alloc.stable(tables, cx))
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_middle::mir::Const<'tcx> {
    type T = stable_mir::ty::MirConst;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        let id = tables.intern_mir_const(cx.lift(*self).unwrap());
        match *self {
            mir::Const::Ty(ty, c) => MirConst::new(
                stable_mir::ty::ConstantKind::Ty(c.stable(tables, cx)),
                ty.stable(tables, cx),
                id,
            ),
            mir::Const::Unevaluated(unev_const, ty) => {
                let kind =
                    stable_mir::ty::ConstantKind::Unevaluated(stable_mir::ty::UnevaluatedConst {
                        def: tables.const_def(unev_const.def),
                        args: unev_const.args.stable(tables, cx),
                        promoted: unev_const.promoted.map(|u| u.as_u32()),
                    });
                let ty = ty.stable(tables, cx);
                MirConst::new(kind, ty, id)
            }
            mir::Const::Val(mir::ConstValue::ZeroSized, ty) => {
                let ty = ty.stable(tables, cx);
                MirConst::new(ConstantKind::ZeroSized, ty, id)
            }
            mir::Const::Val(val, ty) => {
                let ty = cx.lift(ty).unwrap();
                let val = cx.lift(val).unwrap();
                let kind = ConstantKind::Allocated(alloc::new_allocation(ty, val, tables, cx));
                let ty = ty.stable(tables, cx);
                MirConst::new(kind, ty, id)
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::interpret::ErrorHandled {
    type T = Error;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &SmirCtxt<'_, BridgeTys>) -> Self::T {
        Error::new(format!("{self:?}"))
    }
}

impl<'tcx> Stable<'tcx> for MonoItem<'tcx> {
    type T = stable_mir::mir::mono::MonoItem;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &SmirCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use stable_mir::mir::mono::MonoItem as StableMonoItem;
        match self {
            MonoItem::Fn(instance) => StableMonoItem::Fn(instance.stable(tables, cx)),
            MonoItem::Static(def_id) => StableMonoItem::Static(tables.static_def(*def_id)),
            MonoItem::GlobalAsm(item_id) => StableMonoItem::GlobalAsm(opaque(item_id)),
        }
    }
}
