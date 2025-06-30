//! Conversion of internal Rust compiler `mir` items to stable ones.

use rustc_middle::mir::interpret::alloc_range;
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::{bug, mir};
use stable_mir::mir::alloc::GlobalAlloc;
use stable_mir::mir::{ConstOperand, Statement, UserTypeProjection, VarDebugInfoFragment};
use stable_mir::ty::{Allocation, ConstantKind, MirConst};
use stable_mir::{Error, opaque};

use crate::rustc_smir::{Stable, Tables, alloc};
use crate::stable_mir;

impl<'tcx> Stable<'tcx> for mir::Body<'tcx> {
    type T = stable_mir::mir::Body;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        stable_mir::mir::Body::new(
            self.basic_blocks
                .iter()
                .map(|block| stable_mir::mir::BasicBlock {
                    terminator: block.terminator().stable(tables),
                    statements: block
                        .statements
                        .iter()
                        .map(|statement| statement.stable(tables))
                        .collect(),
                })
                .collect(),
            self.local_decls
                .iter()
                .map(|decl| stable_mir::mir::LocalDecl {
                    ty: decl.ty.stable(tables),
                    span: decl.source_info.span.stable(tables),
                    mutability: decl.mutability.stable(tables),
                })
                .collect(),
            self.arg_count,
            self.var_debug_info.iter().map(|info| info.stable(tables)).collect(),
            self.spread_arg.stable(tables),
            self.span.stable(tables),
        )
    }
}

impl<'tcx> Stable<'tcx> for mir::VarDebugInfo<'tcx> {
    type T = stable_mir::mir::VarDebugInfo;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        stable_mir::mir::VarDebugInfo {
            name: self.name.to_string(),
            source_info: self.source_info.stable(tables),
            composite: self.composite.as_ref().map(|composite| composite.stable(tables)),
            value: self.value.stable(tables),
            argument_index: self.argument_index,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::Statement<'tcx> {
    type T = stable_mir::mir::Statement;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        Statement { kind: self.kind.stable(tables), span: self.source_info.span.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for mir::SourceInfo {
    type T = stable_mir::mir::SourceInfo;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        stable_mir::mir::SourceInfo { span: self.span.stable(tables), scope: self.scope.into() }
    }
}

impl<'tcx> Stable<'tcx> for mir::VarDebugInfoFragment<'tcx> {
    type T = stable_mir::mir::VarDebugInfoFragment;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        VarDebugInfoFragment {
            ty: self.ty.stable(tables),
            projection: self.projection.iter().map(|e| e.stable(tables)).collect(),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::VarDebugInfoContents<'tcx> {
    type T = stable_mir::mir::VarDebugInfoContents;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        match self {
            mir::VarDebugInfoContents::Place(place) => {
                stable_mir::mir::VarDebugInfoContents::Place(place.stable(tables))
            }
            mir::VarDebugInfoContents::Const(const_operand) => {
                let op = ConstOperand {
                    span: const_operand.span.stable(tables),
                    user_ty: const_operand.user_ty.map(|index| index.as_usize()),
                    const_: const_operand.const_.stable(tables),
                };
                stable_mir::mir::VarDebugInfoContents::Const(op)
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::StatementKind<'tcx> {
    type T = stable_mir::mir::StatementKind;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        match self {
            mir::StatementKind::Assign(assign) => stable_mir::mir::StatementKind::Assign(
                assign.0.stable(tables),
                assign.1.stable(tables),
            ),
            mir::StatementKind::FakeRead(fake_read_place) => {
                stable_mir::mir::StatementKind::FakeRead(
                    fake_read_place.0.stable(tables),
                    fake_read_place.1.stable(tables),
                )
            }
            mir::StatementKind::SetDiscriminant { place, variant_index } => {
                stable_mir::mir::StatementKind::SetDiscriminant {
                    place: place.as_ref().stable(tables),
                    variant_index: variant_index.stable(tables),
                }
            }
            mir::StatementKind::Deinit(place) => {
                stable_mir::mir::StatementKind::Deinit(place.stable(tables))
            }

            mir::StatementKind::StorageLive(place) => {
                stable_mir::mir::StatementKind::StorageLive(place.stable(tables))
            }

            mir::StatementKind::StorageDead(place) => {
                stable_mir::mir::StatementKind::StorageDead(place.stable(tables))
            }
            mir::StatementKind::Retag(retag, place) => {
                stable_mir::mir::StatementKind::Retag(retag.stable(tables), place.stable(tables))
            }
            mir::StatementKind::PlaceMention(place) => {
                stable_mir::mir::StatementKind::PlaceMention(place.stable(tables))
            }
            mir::StatementKind::AscribeUserType(place_projection, variance) => {
                stable_mir::mir::StatementKind::AscribeUserType {
                    place: place_projection.as_ref().0.stable(tables),
                    projections: place_projection.as_ref().1.stable(tables),
                    variance: variance.stable(tables),
                }
            }
            mir::StatementKind::Coverage(coverage) => {
                stable_mir::mir::StatementKind::Coverage(opaque(coverage))
            }
            mir::StatementKind::Intrinsic(intrinstic) => {
                stable_mir::mir::StatementKind::Intrinsic(intrinstic.stable(tables))
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
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use rustc_middle::mir::Rvalue::*;
        match self {
            Use(op) => stable_mir::mir::Rvalue::Use(op.stable(tables)),
            Repeat(op, len) => {
                let len = len.stable(tables);
                stable_mir::mir::Rvalue::Repeat(op.stable(tables), len)
            }
            Ref(region, kind, place) => stable_mir::mir::Rvalue::Ref(
                region.stable(tables),
                kind.stable(tables),
                place.stable(tables),
            ),
            ThreadLocalRef(def_id) => {
                stable_mir::mir::Rvalue::ThreadLocalRef(tables.crate_item(*def_id))
            }
            RawPtr(mutability, place) => {
                stable_mir::mir::Rvalue::AddressOf(mutability.stable(tables), place.stable(tables))
            }
            Len(place) => stable_mir::mir::Rvalue::Len(place.stable(tables)),
            Cast(cast_kind, op, ty) => stable_mir::mir::Rvalue::Cast(
                cast_kind.stable(tables),
                op.stable(tables),
                ty.stable(tables),
            ),
            BinaryOp(bin_op, ops) => {
                if let Some(bin_op) = bin_op.overflowing_to_wrapping() {
                    stable_mir::mir::Rvalue::CheckedBinaryOp(
                        bin_op.stable(tables),
                        ops.0.stable(tables),
                        ops.1.stable(tables),
                    )
                } else {
                    stable_mir::mir::Rvalue::BinaryOp(
                        bin_op.stable(tables),
                        ops.0.stable(tables),
                        ops.1.stable(tables),
                    )
                }
            }
            NullaryOp(null_op, ty) => {
                stable_mir::mir::Rvalue::NullaryOp(null_op.stable(tables), ty.stable(tables))
            }
            UnaryOp(un_op, op) => {
                stable_mir::mir::Rvalue::UnaryOp(un_op.stable(tables), op.stable(tables))
            }
            Discriminant(place) => stable_mir::mir::Rvalue::Discriminant(place.stable(tables)),
            Aggregate(agg_kind, operands) => {
                let operands = operands.iter().map(|op| op.stable(tables)).collect();
                stable_mir::mir::Rvalue::Aggregate(agg_kind.stable(tables), operands)
            }
            ShallowInitBox(op, ty) => {
                stable_mir::mir::Rvalue::ShallowInitBox(op.stable(tables), ty.stable(tables))
            }
            CopyForDeref(place) => stable_mir::mir::Rvalue::CopyForDeref(place.stable(tables)),
            WrapUnsafeBinder(..) => todo!("FIXME(unsafe_binders):"),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::Mutability {
    type T = stable_mir::mir::Mutability;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        use rustc_hir::Mutability::*;
        match *self {
            Not => stable_mir::mir::Mutability::Not,
            Mut => stable_mir::mir::Mutability::Mut,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::RawPtrKind {
    type T = stable_mir::mir::RawPtrKind;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
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
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use rustc_middle::mir::BorrowKind::*;
        match *self {
            Shared => stable_mir::mir::BorrowKind::Shared,
            Fake(kind) => stable_mir::mir::BorrowKind::Fake(kind.stable(tables)),
            Mut { kind } => stable_mir::mir::BorrowKind::Mut { kind: kind.stable(tables) },
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::MutBorrowKind {
    type T = stable_mir::mir::MutBorrowKind;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
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
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        use rustc_middle::mir::FakeBorrowKind::*;
        match *self {
            Deep => stable_mir::mir::FakeBorrowKind::Deep,
            Shallow => stable_mir::mir::FakeBorrowKind::Shallow,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::NullOp<'tcx> {
    type T = stable_mir::mir::NullOp;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use rustc_middle::mir::NullOp::*;
        match self {
            SizeOf => stable_mir::mir::NullOp::SizeOf,
            AlignOf => stable_mir::mir::NullOp::AlignOf,
            OffsetOf(indices) => stable_mir::mir::NullOp::OffsetOf(
                indices.iter().map(|idx| idx.stable(tables)).collect(),
            ),
            UbChecks => stable_mir::mir::NullOp::UbChecks,
            ContractChecks => stable_mir::mir::NullOp::ContractChecks,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::CastKind {
    type T = stable_mir::mir::CastKind;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use rustc_middle::mir::CastKind::*;
        use rustc_middle::ty::adjustment::PointerCoercion;
        match self {
            PointerExposeProvenance => stable_mir::mir::CastKind::PointerExposeAddress,
            PointerWithExposedProvenance => stable_mir::mir::CastKind::PointerWithExposedProvenance,
            PointerCoercion(PointerCoercion::DynStar, _) => stable_mir::mir::CastKind::DynStar,
            PointerCoercion(c, _) => stable_mir::mir::CastKind::PointerCoercion(c.stable(tables)),
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
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
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
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use rustc_middle::mir::Operand::*;
        match self {
            Copy(place) => stable_mir::mir::Operand::Copy(place.stable(tables)),
            Move(place) => stable_mir::mir::Operand::Move(place.stable(tables)),
            Constant(c) => stable_mir::mir::Operand::Constant(c.stable(tables)),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::ConstOperand<'tcx> {
    type T = stable_mir::mir::ConstOperand;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        stable_mir::mir::ConstOperand {
            span: self.span.stable(tables),
            user_ty: self.user_ty.map(|u| u.as_usize()).or(None),
            const_: self.const_.stable(tables),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::Place<'tcx> {
    type T = stable_mir::mir::Place;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        stable_mir::mir::Place {
            local: self.local.as_usize(),
            projection: self.projection.iter().map(|e| e.stable(tables)).collect(),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::PlaceElem<'tcx> {
    type T = stable_mir::mir::ProjectionElem;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use rustc_middle::mir::ProjectionElem::*;
        match self {
            Deref => stable_mir::mir::ProjectionElem::Deref,
            Field(idx, ty) => {
                stable_mir::mir::ProjectionElem::Field(idx.stable(tables), ty.stable(tables))
            }
            Index(local) => stable_mir::mir::ProjectionElem::Index(local.stable(tables)),
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
            Downcast(_, idx) => stable_mir::mir::ProjectionElem::Downcast(idx.stable(tables)),
            OpaqueCast(ty) => stable_mir::mir::ProjectionElem::OpaqueCast(ty.stable(tables)),
            Subtype(ty) => stable_mir::mir::ProjectionElem::Subtype(ty.stable(tables)),
            UnwrapUnsafeBinder(..) => todo!("FIXME(unsafe_binders):"),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::UserTypeProjection {
    type T = stable_mir::mir::UserTypeProjection;

    fn stable(&self, _tables: &mut Tables<'_>) -> Self::T {
        UserTypeProjection { base: self.base.as_usize(), projection: opaque(&self.projs) }
    }
}

impl<'tcx> Stable<'tcx> for mir::Local {
    type T = stable_mir::mir::Local;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        self.as_usize()
    }
}

impl<'tcx> Stable<'tcx> for mir::RetagKind {
    type T = stable_mir::mir::RetagKind;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
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
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
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

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use rustc_middle::mir::NonDivergingIntrinsic;
        use stable_mir::mir::CopyNonOverlapping;
        match self {
            NonDivergingIntrinsic::Assume(op) => {
                stable_mir::mir::NonDivergingIntrinsic::Assume(op.stable(tables))
            }
            NonDivergingIntrinsic::CopyNonOverlapping(copy_non_overlapping) => {
                stable_mir::mir::NonDivergingIntrinsic::CopyNonOverlapping(CopyNonOverlapping {
                    src: copy_non_overlapping.src.stable(tables),
                    dst: copy_non_overlapping.dst.stable(tables),
                    count: copy_non_overlapping.count.stable(tables),
                })
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::AssertMessage<'tcx> {
    type T = stable_mir::mir::AssertMessage;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use rustc_middle::mir::AssertKind;
        match self {
            AssertKind::BoundsCheck { len, index } => stable_mir::mir::AssertMessage::BoundsCheck {
                len: len.stable(tables),
                index: index.stable(tables),
            },
            AssertKind::Overflow(bin_op, op1, op2) => stable_mir::mir::AssertMessage::Overflow(
                bin_op.stable(tables),
                op1.stable(tables),
                op2.stable(tables),
            ),
            AssertKind::OverflowNeg(op) => {
                stable_mir::mir::AssertMessage::OverflowNeg(op.stable(tables))
            }
            AssertKind::DivisionByZero(op) => {
                stable_mir::mir::AssertMessage::DivisionByZero(op.stable(tables))
            }
            AssertKind::RemainderByZero(op) => {
                stable_mir::mir::AssertMessage::RemainderByZero(op.stable(tables))
            }
            AssertKind::ResumedAfterReturn(coroutine) => {
                stable_mir::mir::AssertMessage::ResumedAfterReturn(coroutine.stable(tables))
            }
            AssertKind::ResumedAfterPanic(coroutine) => {
                stable_mir::mir::AssertMessage::ResumedAfterPanic(coroutine.stable(tables))
            }
            AssertKind::ResumedAfterDrop(coroutine) => {
                stable_mir::mir::AssertMessage::ResumedAfterDrop(coroutine.stable(tables))
            }
            AssertKind::MisalignedPointerDereference { required, found } => {
                stable_mir::mir::AssertMessage::MisalignedPointerDereference {
                    required: required.stable(tables),
                    found: found.stable(tables),
                }
            }
            AssertKind::NullPointerDereference => {
                stable_mir::mir::AssertMessage::NullPointerDereference
            }
            AssertKind::InvalidEnumConstruction(source) => {
                stable_mir::mir::AssertMessage::InvalidEnumConstruction(source.stable(tables))
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::BinOp {
    type T = stable_mir::mir::BinOp;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
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
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
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
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        match self {
            mir::AggregateKind::Array(ty) => {
                stable_mir::mir::AggregateKind::Array(ty.stable(tables))
            }
            mir::AggregateKind::Tuple => stable_mir::mir::AggregateKind::Tuple,
            mir::AggregateKind::Adt(def_id, var_idx, generic_arg, user_ty_index, field_idx) => {
                stable_mir::mir::AggregateKind::Adt(
                    tables.adt_def(*def_id),
                    var_idx.stable(tables),
                    generic_arg.stable(tables),
                    user_ty_index.map(|idx| idx.index()),
                    field_idx.map(|idx| idx.index()),
                )
            }
            mir::AggregateKind::Closure(def_id, generic_arg) => {
                stable_mir::mir::AggregateKind::Closure(
                    tables.closure_def(*def_id),
                    generic_arg.stable(tables),
                )
            }
            mir::AggregateKind::Coroutine(def_id, generic_arg) => {
                stable_mir::mir::AggregateKind::Coroutine(
                    tables.coroutine_def(*def_id),
                    generic_arg.stable(tables),
                    tables.tcx.coroutine_movability(*def_id).stable(tables),
                )
            }
            mir::AggregateKind::CoroutineClosure(def_id, generic_args) => {
                stable_mir::mir::AggregateKind::CoroutineClosure(
                    tables.coroutine_closure_def(*def_id),
                    generic_args.stable(tables),
                )
            }
            mir::AggregateKind::RawPtr(ty, mutability) => {
                stable_mir::mir::AggregateKind::RawPtr(ty.stable(tables), mutability.stable(tables))
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::InlineAsmOperand<'tcx> {
    type T = stable_mir::mir::InlineAsmOperand;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use rustc_middle::mir::InlineAsmOperand;

        let (in_value, out_place) = match self {
            InlineAsmOperand::In { value, .. } => (Some(value.stable(tables)), None),
            InlineAsmOperand::Out { place, .. } => (None, place.map(|place| place.stable(tables))),
            InlineAsmOperand::InOut { in_value, out_place, .. } => {
                (Some(in_value.stable(tables)), out_place.map(|place| place.stable(tables)))
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
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::mir::Terminator;
        Terminator { kind: self.kind.stable(tables), span: self.source_info.span.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for mir::TerminatorKind<'tcx> {
    type T = stable_mir::mir::TerminatorKind;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::mir::TerminatorKind;
        match self {
            mir::TerminatorKind::Goto { target } => {
                TerminatorKind::Goto { target: target.as_usize() }
            }
            mir::TerminatorKind::SwitchInt { discr, targets } => TerminatorKind::SwitchInt {
                discr: discr.stable(tables),
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
                place: place.stable(tables),
                target: target.as_usize(),
                unwind: unwind.stable(tables),
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
                func: func.stable(tables),
                args: args.iter().map(|arg| arg.node.stable(tables)).collect(),
                destination: destination.stable(tables),
                target: target.map(|t| t.as_usize()),
                unwind: unwind.stable(tables),
            },
            mir::TerminatorKind::TailCall { func: _, args: _, fn_span: _ } => todo!(),
            mir::TerminatorKind::Assert { cond, expected, msg, target, unwind } => {
                TerminatorKind::Assert {
                    cond: cond.stable(tables),
                    expected: *expected,
                    msg: msg.stable(tables),
                    target: target.as_usize(),
                    unwind: unwind.stable(tables),
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
                operands: operands.iter().map(|operand| operand.stable(tables)).collect(),
                options: format!("{options:?}"),
                line_spans: format!("{line_spans:?}"),
                // FIXME: Figure out how to do labels in SMIR
                destination: targets.first().map(|d| d.as_usize()),
                unwind: unwind.stable(tables),
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

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        self.inner().stable(tables)
    }
}

impl<'tcx> Stable<'tcx> for mir::interpret::Allocation {
    type T = stable_mir::ty::Allocation;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        alloc::allocation_filter(self, alloc_range(rustc_abi::Size::ZERO, self.size()), tables)
    }
}

impl<'tcx> Stable<'tcx> for mir::interpret::AllocId {
    type T = stable_mir::mir::alloc::AllocId;
    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        tables.create_alloc_id(*self)
    }
}

impl<'tcx> Stable<'tcx> for mir::interpret::GlobalAlloc<'tcx> {
    type T = GlobalAlloc;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        match self {
            mir::interpret::GlobalAlloc::Function { instance, .. } => {
                GlobalAlloc::Function(instance.stable(tables))
            }
            mir::interpret::GlobalAlloc::VTable(ty, dyn_ty) => {
                // FIXME: Should we record the whole vtable?
                GlobalAlloc::VTable(ty.stable(tables), dyn_ty.principal().stable(tables))
            }
            mir::interpret::GlobalAlloc::Static(def) => {
                GlobalAlloc::Static(tables.static_def(*def))
            }
            mir::interpret::GlobalAlloc::Memory(alloc) => GlobalAlloc::Memory(alloc.stable(tables)),
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_middle::mir::Const<'tcx> {
    type T = stable_mir::ty::MirConst;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        let id = tables.intern_mir_const(tables.tcx.lift(*self).unwrap());
        match *self {
            mir::Const::Ty(ty, c) => MirConst::new(
                stable_mir::ty::ConstantKind::Ty(c.stable(tables)),
                ty.stable(tables),
                id,
            ),
            mir::Const::Unevaluated(unev_const, ty) => {
                let kind =
                    stable_mir::ty::ConstantKind::Unevaluated(stable_mir::ty::UnevaluatedConst {
                        def: tables.const_def(unev_const.def),
                        args: unev_const.args.stable(tables),
                        promoted: unev_const.promoted.map(|u| u.as_u32()),
                    });
                let ty = ty.stable(tables);
                MirConst::new(kind, ty, id)
            }
            mir::Const::Val(mir::ConstValue::ZeroSized, ty) => {
                let ty = ty.stable(tables);
                MirConst::new(ConstantKind::ZeroSized, ty, id)
            }
            mir::Const::Val(val, ty) => {
                let ty = tables.tcx.lift(ty).unwrap();
                let val = tables.tcx.lift(val).unwrap();
                let kind = ConstantKind::Allocated(alloc::new_allocation(ty, val, tables));
                let ty = ty.stable(tables);
                MirConst::new(kind, ty, id)
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::interpret::ErrorHandled {
    type T = Error;

    fn stable(&self, _tables: &mut Tables<'_>) -> Self::T {
        Error::new(format!("{self:?}"))
    }
}

impl<'tcx> Stable<'tcx> for MonoItem<'tcx> {
    type T = stable_mir::mir::mono::MonoItem;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        use stable_mir::mir::mono::MonoItem as StableMonoItem;
        match self {
            MonoItem::Fn(instance) => StableMonoItem::Fn(instance.stable(tables)),
            MonoItem::Static(def_id) => StableMonoItem::Static(tables.static_def(*def_id)),
            MonoItem::GlobalAsm(item_id) => StableMonoItem::GlobalAsm(opaque(item_id)),
        }
    }
}
