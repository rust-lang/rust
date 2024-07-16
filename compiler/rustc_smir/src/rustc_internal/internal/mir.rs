use crate::rustc_smir::Tables;
use rustc_middle::ty::{self as rustc_ty, TyCtxt};
use stable_mir::mir::alloc::AllocId;
use stable_mir::mir::mono::{Instance, MonoItem, StaticDef};
use stable_mir::mir::{
    AggregateKind, AssertMessage, BinOp, Body, BorrowKind, CastKind, ConstOperand,
    CopyNonOverlapping, CoroutineDesugaring, CoroutineKind, CoroutineSource, FakeBorrowKind,
    FakeReadCause, LocalDecl, MutBorrowKind, Mutability, NonDivergingIntrinsic, NullOp, Operand,
    Place, PointerCoercion, ProjectionElem, RetagKind, Rvalue, Safety, Statement, StatementKind,
    SwitchTargets, Terminator, TerminatorKind, UnOp, UnwindAction, UserTypeProjection, Variance,
};

use super::RustcInternal;

impl RustcInternal for Mutability {
    type T<'tcx> = rustc_ty::Mutability;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            Mutability::Not => rustc_ty::Mutability::Not,
            Mutability::Mut => rustc_ty::Mutability::Mut,
        }
    }
}

impl RustcInternal for MonoItem {
    type T<'tcx> = rustc_middle::mir::mono::MonoItem<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        use rustc_middle::mir::mono as rustc_mono;
        match self {
            MonoItem::Fn(instance) => rustc_mono::MonoItem::Fn(instance.internal(tables, tcx)),
            MonoItem::Static(def) => rustc_mono::MonoItem::Static(def.internal(tables, tcx)),
            MonoItem::GlobalAsm(_) => {
                unimplemented!()
            }
        }
    }
}

impl RustcInternal for Instance {
    type T<'tcx> = rustc_ty::Instance<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        tcx.lift(tables.instances[self.def]).unwrap()
    }
}

impl RustcInternal for StaticDef {
    type T<'tcx> = rustc_span::def_id::DefId;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        self.0.internal(tables, tcx)
    }
}

impl RustcInternal for AllocId {
    type T<'tcx> = rustc_middle::mir::interpret::AllocId;
    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        tcx.lift(tables.alloc_ids[*self]).unwrap()
    }
}

impl RustcInternal for Safety {
    type T<'tcx> = rustc_hir::Safety;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            Safety::Unsafe => rustc_hir::Safety::Unsafe,
            Safety::Safe => rustc_hir::Safety::Safe,
        }
    }
}

impl RustcInternal for Place {
    type T<'tcx> = rustc_middle::mir::Place<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_middle::mir::Place {
            local: rustc_middle::mir::Local::from_usize(self.local),
            projection: tcx.mk_place_elems(&self.projection.internal(tables, tcx)),
        }
    }
}

impl RustcInternal for ProjectionElem {
    type T<'tcx> = rustc_middle::mir::PlaceElem<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            ProjectionElem::Deref => rustc_middle::mir::PlaceElem::Deref,
            ProjectionElem::Field(idx, ty) => {
                rustc_middle::mir::PlaceElem::Field((*idx).into(), ty.internal(tables, tcx))
            }
            ProjectionElem::Index(idx) => rustc_middle::mir::PlaceElem::Index((*idx).into()),
            ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
                rustc_middle::mir::PlaceElem::ConstantIndex {
                    offset: *offset,
                    min_length: *min_length,
                    from_end: *from_end,
                }
            }
            ProjectionElem::Subslice { from, to, from_end } => {
                rustc_middle::mir::PlaceElem::Subslice { from: *from, to: *to, from_end: *from_end }
            }
            ProjectionElem::Downcast(idx) => {
                rustc_middle::mir::PlaceElem::Downcast(None, idx.internal(tables, tcx))
            }
            ProjectionElem::OpaqueCast(ty) => {
                rustc_middle::mir::PlaceElem::OpaqueCast(ty.internal(tables, tcx))
            }
            ProjectionElem::Subtype(ty) => {
                rustc_middle::mir::PlaceElem::Subtype(ty.internal(tables, tcx))
            }
        }
    }
}

impl RustcInternal for BinOp {
    type T<'tcx> = rustc_middle::mir::BinOp;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            BinOp::Add => rustc_middle::mir::BinOp::Add,
            BinOp::AddUnchecked => rustc_middle::mir::BinOp::AddUnchecked,
            BinOp::Sub => rustc_middle::mir::BinOp::Sub,
            BinOp::SubUnchecked => rustc_middle::mir::BinOp::SubUnchecked,
            BinOp::Mul => rustc_middle::mir::BinOp::Mul,
            BinOp::MulUnchecked => rustc_middle::mir::BinOp::MulUnchecked,
            BinOp::Div => rustc_middle::mir::BinOp::Div,
            BinOp::Rem => rustc_middle::mir::BinOp::Rem,
            BinOp::BitXor => rustc_middle::mir::BinOp::BitXor,
            BinOp::BitAnd => rustc_middle::mir::BinOp::BitAnd,
            BinOp::BitOr => rustc_middle::mir::BinOp::BitOr,
            BinOp::Shl => rustc_middle::mir::BinOp::Shl,
            BinOp::ShlUnchecked => rustc_middle::mir::BinOp::ShlUnchecked,
            BinOp::Shr => rustc_middle::mir::BinOp::Shr,
            BinOp::ShrUnchecked => rustc_middle::mir::BinOp::ShrUnchecked,
            BinOp::Eq => rustc_middle::mir::BinOp::Eq,
            BinOp::Lt => rustc_middle::mir::BinOp::Lt,
            BinOp::Le => rustc_middle::mir::BinOp::Le,
            BinOp::Ne => rustc_middle::mir::BinOp::Ne,
            BinOp::Ge => rustc_middle::mir::BinOp::Ge,
            BinOp::Gt => rustc_middle::mir::BinOp::Gt,
            BinOp::Cmp => rustc_middle::mir::BinOp::Cmp,
            BinOp::Offset => rustc_middle::mir::BinOp::Offset,
        }
    }
}

impl RustcInternal for UnOp {
    type T<'tcx> = rustc_middle::mir::UnOp;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            UnOp::Not => rustc_middle::mir::UnOp::Not,
            UnOp::Neg => rustc_middle::mir::UnOp::Neg,
            UnOp::PtrMetadata => rustc_middle::mir::UnOp::PtrMetadata,
        }
    }
}

impl RustcInternal for AggregateKind {
    type T<'tcx> = rustc_middle::mir::AggregateKind<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            AggregateKind::Array(ty) => {
                rustc_middle::mir::AggregateKind::Array(ty.internal(tables, tcx).into())
            }
            AggregateKind::Tuple => rustc_middle::mir::AggregateKind::Tuple,
            AggregateKind::Adt(
                adt_def,
                variant_idx,
                generic_args,
                maybe_user_type_annotation_index,
                maybe_field_idx,
            ) => rustc_middle::mir::AggregateKind::Adt(
                adt_def.0.internal(tables, tcx).into(),
                variant_idx.internal(tables, tcx).into(),
                generic_args.internal(tables, tcx).into(),
                maybe_user_type_annotation_index
                    .map(|idx| rustc_middle::ty::UserTypeAnnotationIndex::from_usize(idx)),
                maybe_field_idx.map(|idx| rustc_target::abi::FieldIdx::from_usize(idx)),
            ),
            AggregateKind::Closure(closure_def, generic_args) => {
                rustc_middle::mir::AggregateKind::Closure(
                    closure_def.0.internal(tables, tcx).into(),
                    generic_args.internal(tables, tcx).into(),
                )
            }
            AggregateKind::Coroutine(coroutine_def, generic_args, _) => {
                rustc_middle::mir::AggregateKind::Coroutine(
                    coroutine_def.0.internal(tables, tcx).into(),
                    generic_args.internal(tables, tcx).into(),
                )
            }
            AggregateKind::RawPtr(ty, mutability) => rustc_middle::mir::AggregateKind::RawPtr(
                ty.internal(tables, tcx).into(),
                mutability.internal(tables, tcx).into(),
            ),
        }
    }
}

impl RustcInternal for ConstOperand {
    type T<'tcx> = rustc_middle::mir::ConstOperand<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_middle::mir::ConstOperand {
            span: self.span.internal(tables, tcx).into(),
            user_ty: self.user_ty.map(|idx| rustc_ty::UserTypeAnnotationIndex::from_usize(idx)),
            const_: self.const_.internal(tables, tcx).into(),
        }
    }
}

impl RustcInternal for Operand {
    type T<'tcx> = rustc_middle::mir::Operand<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            Operand::Copy(place) => {
                rustc_middle::mir::Operand::Copy(place.internal(tables, tcx).into())
            }
            Operand::Move(place) => {
                rustc_middle::mir::Operand::Move(place.internal(tables, tcx).into())
            }
            Operand::Constant(const_operand) => rustc_middle::mir::Operand::Constant(Box::new(
                const_operand.internal(tables, tcx).into(),
            )),
        }
    }
}

impl RustcInternal for PointerCoercion {
    type T<'tcx> = rustc_middle::ty::adjustment::PointerCoercion;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            PointerCoercion::ReifyFnPointer => {
                rustc_middle::ty::adjustment::PointerCoercion::ReifyFnPointer
            }
            PointerCoercion::UnsafeFnPointer => {
                rustc_middle::ty::adjustment::PointerCoercion::UnsafeFnPointer
            }
            PointerCoercion::ClosureFnPointer(safety) => {
                rustc_middle::ty::adjustment::PointerCoercion::ClosureFnPointer(
                    safety.internal(tables, tcx).into(),
                )
            }
            PointerCoercion::MutToConstPointer => {
                rustc_middle::ty::adjustment::PointerCoercion::MutToConstPointer
            }
            PointerCoercion::ArrayToPointer => {
                rustc_middle::ty::adjustment::PointerCoercion::ArrayToPointer
            }
            PointerCoercion::Unsize => rustc_middle::ty::adjustment::PointerCoercion::Unsize,
        }
    }
}

impl RustcInternal for CastKind {
    type T<'tcx> = rustc_middle::mir::CastKind;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            CastKind::PointerExposeAddress => rustc_middle::mir::CastKind::PointerExposeProvenance,
            CastKind::PointerWithExposedProvenance => {
                rustc_middle::mir::CastKind::PointerWithExposedProvenance
            }
            CastKind::PointerCoercion(ptr_coercion) => {
                rustc_middle::mir::CastKind::PointerCoercion(
                    ptr_coercion.internal(tables, tcx).into(),
                )
            }
            CastKind::DynStar => rustc_middle::mir::CastKind::DynStar,
            CastKind::IntToInt => rustc_middle::mir::CastKind::IntToInt,
            CastKind::FloatToInt => rustc_middle::mir::CastKind::FloatToInt,
            CastKind::FloatToFloat => rustc_middle::mir::CastKind::FloatToFloat,
            CastKind::IntToFloat => rustc_middle::mir::CastKind::IntToFloat,
            CastKind::PtrToPtr => rustc_middle::mir::CastKind::PtrToPtr,
            CastKind::FnPtrToPtr => rustc_middle::mir::CastKind::FnPtrToPtr,
            CastKind::Transmute => rustc_middle::mir::CastKind::Transmute,
        }
    }
}

impl RustcInternal for FakeBorrowKind {
    type T<'tcx> = rustc_middle::mir::FakeBorrowKind;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            FakeBorrowKind::Deep => rustc_middle::mir::FakeBorrowKind::Deep,
            FakeBorrowKind::Shallow => rustc_middle::mir::FakeBorrowKind::Shallow,
        }
    }
}

impl RustcInternal for MutBorrowKind {
    type T<'tcx> = rustc_middle::mir::MutBorrowKind;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            MutBorrowKind::Default => rustc_middle::mir::MutBorrowKind::Default,
            MutBorrowKind::TwoPhaseBorrow => rustc_middle::mir::MutBorrowKind::TwoPhaseBorrow,
            MutBorrowKind::ClosureCapture => rustc_middle::mir::MutBorrowKind::ClosureCapture,
        }
    }
}

impl RustcInternal for BorrowKind {
    type T<'tcx> = rustc_middle::mir::BorrowKind;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            BorrowKind::Shared => rustc_middle::mir::BorrowKind::Shared,
            BorrowKind::Fake(fake_borrow_kind) => {
                rustc_middle::mir::BorrowKind::Fake(fake_borrow_kind.internal(tables, tcx).into())
            }
            BorrowKind::Mut { kind } => {
                rustc_middle::mir::BorrowKind::Mut { kind: kind.internal(tables, tcx).into() }
            }
        }
    }
}

impl RustcInternal for NullOp {
    type T<'tcx> = rustc_middle::mir::NullOp<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            NullOp::SizeOf => rustc_middle::mir::NullOp::SizeOf,
            NullOp::AlignOf => rustc_middle::mir::NullOp::AlignOf,
            NullOp::OffsetOf(offsets) => rustc_middle::mir::NullOp::OffsetOf(
                tcx.mk_offset_of(
                    offsets
                        .iter()
                        .map(|(variant_idx, field_idx)| {
                            (
                                variant_idx.internal(tables, tcx),
                                rustc_target::abi::FieldIdx::from_usize(*field_idx),
                            )
                        })
                        .collect::<Vec<_>>()
                        .as_slice(),
                ),
            ),
            NullOp::UbChecks => rustc_middle::mir::NullOp::UbChecks,
        }
    }
}

impl RustcInternal for Rvalue {
    type T<'tcx> = rustc_middle::mir::Rvalue<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            Rvalue::AddressOf(mutability, place) => rustc_middle::mir::Rvalue::AddressOf(
                mutability.internal(tables, tcx).into(),
                place.internal(tables, tcx).into(),
            ),
            Rvalue::Aggregate(aggregate_kind, operands) => rustc_middle::mir::Rvalue::Aggregate(
                Box::new(aggregate_kind.internal(tables, tcx).into()),
                rustc_index::IndexVec::from_raw(
                    operands.iter().map(|operand| operand.internal(tables, tcx).into()).collect(),
                ),
            ),
            Rvalue::BinaryOp(bin_op, left_operand, right_operand)
            | Rvalue::CheckedBinaryOp(bin_op, left_operand, right_operand) => {
                rustc_middle::mir::Rvalue::BinaryOp(
                    bin_op.internal(tables, tcx).into(),
                    Box::new((
                        left_operand.internal(tables, tcx).into(),
                        right_operand.internal(tables, tcx).into(),
                    )),
                )
            }
            Rvalue::Cast(cast_kind, operand, ty) => rustc_middle::mir::Rvalue::Cast(
                cast_kind.internal(tables, tcx).into(),
                operand.internal(tables, tcx).into(),
                ty.internal(tables, tcx).into(),
            ),
            Rvalue::CopyForDeref(place) => {
                rustc_middle::mir::Rvalue::CopyForDeref(place.internal(tables, tcx).into())
            }
            Rvalue::Discriminant(place) => {
                rustc_middle::mir::Rvalue::Discriminant(place.internal(tables, tcx).into())
            }
            Rvalue::Len(place) => {
                rustc_middle::mir::Rvalue::Len(place.internal(tables, tcx).into())
            }
            Rvalue::Ref(region, borrow_kind, place) => rustc_middle::mir::Rvalue::Ref(
                region.internal(tables, tcx).into(),
                borrow_kind.internal(tables, tcx).into(),
                place.internal(tables, tcx).into(),
            ),
            Rvalue::Repeat(operand, ty_const) => rustc_middle::mir::Rvalue::Repeat(
                operand.internal(tables, tcx).into(),
                ty_const.internal(tables, tcx).into(),
            ),
            Rvalue::ShallowInitBox(operand, ty) => rustc_middle::mir::Rvalue::ShallowInitBox(
                operand.internal(tables, tcx).into(),
                ty.internal(tables, tcx).into(),
            ),
            Rvalue::ThreadLocalRef(crate_item) => {
                rustc_middle::mir::Rvalue::ThreadLocalRef(crate_item.0.internal(tables, tcx).into())
            }
            Rvalue::NullaryOp(null_op, ty) => rustc_middle::mir::Rvalue::NullaryOp(
                null_op.internal(tables, tcx).into(),
                ty.internal(tables, tcx).into(),
            ),
            Rvalue::UnaryOp(un_op, operand) => rustc_middle::mir::Rvalue::UnaryOp(
                un_op.internal(tables, tcx).into(),
                operand.internal(tables, tcx).into(),
            ),
            Rvalue::Use(operand) => {
                rustc_middle::mir::Rvalue::Use(operand.internal(tables, tcx).into())
            }
        }
    }
}

impl RustcInternal for FakeReadCause {
    type T<'tcx> = rustc_middle::mir::FakeReadCause;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            FakeReadCause::ForMatchGuard => rustc_middle::mir::FakeReadCause::ForMatchGuard,
            FakeReadCause::ForMatchedPlace(_opaque) => {
                unimplemented!("cannot convert back from an opaque field")
            }
            FakeReadCause::ForGuardBinding => rustc_middle::mir::FakeReadCause::ForGuardBinding,
            FakeReadCause::ForLet(_opaque) => {
                unimplemented!("cannot convert back from an opaque field")
            }
            FakeReadCause::ForIndex => rustc_middle::mir::FakeReadCause::ForIndex,
        }
    }
}

impl RustcInternal for RetagKind {
    type T<'tcx> = rustc_middle::mir::RetagKind;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            RetagKind::FnEntry => rustc_middle::mir::RetagKind::FnEntry,
            RetagKind::TwoPhase => rustc_middle::mir::RetagKind::TwoPhase,
            RetagKind::Raw => rustc_middle::mir::RetagKind::Raw,
            RetagKind::Default => rustc_middle::mir::RetagKind::Default,
        }
    }
}

impl RustcInternal for UserTypeProjection {
    type T<'tcx> = rustc_middle::mir::UserTypeProjection;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        unimplemented!("cannot convert back from an opaque field")
    }
}

impl RustcInternal for Variance {
    type T<'tcx> = rustc_middle::ty::Variance;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            Variance::Covariant => rustc_middle::ty::Variance::Covariant,
            Variance::Invariant => rustc_middle::ty::Variance::Invariant,
            Variance::Contravariant => rustc_middle::ty::Variance::Contravariant,
            Variance::Bivariant => rustc_middle::ty::Variance::Bivariant,
        }
    }
}

impl RustcInternal for CopyNonOverlapping {
    type T<'tcx> = rustc_middle::mir::CopyNonOverlapping<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_middle::mir::CopyNonOverlapping {
            src: self.src.internal(tables, tcx).into(),
            dst: self.dst.internal(tables, tcx).into(),
            count: self.count.internal(tables, tcx).into(),
        }
    }
}

impl RustcInternal for NonDivergingIntrinsic {
    type T<'tcx> = rustc_middle::mir::NonDivergingIntrinsic<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            NonDivergingIntrinsic::Assume(operand) => {
                rustc_middle::mir::NonDivergingIntrinsic::Assume(
                    operand.internal(tables, tcx).into(),
                )
            }
            NonDivergingIntrinsic::CopyNonOverlapping(copy_non_overlapping) => {
                rustc_middle::mir::NonDivergingIntrinsic::CopyNonOverlapping(
                    copy_non_overlapping.internal(tables, tcx).into(),
                )
            }
        }
    }
}

impl RustcInternal for StatementKind {
    type T<'tcx> = rustc_middle::mir::StatementKind<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            StatementKind::Assign(place, rvalue) => rustc_middle::mir::StatementKind::Assign(
                Box::new((place.internal(tables, tcx).into(), rvalue.internal(tables, tcx).into())),
            ),
            StatementKind::FakeRead(fake_read_cause, place) => {
                rustc_middle::mir::StatementKind::FakeRead(Box::new((
                    fake_read_cause.internal(tables, tcx).into(),
                    place.internal(tables, tcx).into(),
                )))
            }
            StatementKind::SetDiscriminant { place, variant_index } => {
                rustc_middle::mir::StatementKind::SetDiscriminant {
                    place: place.internal(tables, tcx).into(),
                    variant_index: variant_index.internal(tables, tcx).into(),
                }
            }
            StatementKind::Deinit(place) => {
                rustc_middle::mir::StatementKind::Deinit(place.internal(tables, tcx).into())
            }
            StatementKind::StorageLive(local) => rustc_middle::mir::StatementKind::StorageLive(
                rustc_middle::mir::Local::from_usize(*local),
            ),
            StatementKind::StorageDead(local) => rustc_middle::mir::StatementKind::StorageDead(
                rustc_middle::mir::Local::from_usize(*local),
            ),
            StatementKind::Retag(retag_kind, place) => rustc_middle::mir::StatementKind::Retag(
                retag_kind.internal(tables, tcx).into(),
                place.internal(tables, tcx).into(),
            ),
            StatementKind::PlaceMention(place) => rustc_middle::mir::StatementKind::PlaceMention(
                Box::new(place.internal(tables, tcx).into()),
            ),
            StatementKind::AscribeUserType { place, projections, variance } => {
                rustc_middle::mir::StatementKind::AscribeUserType(
                    Box::new((
                        place.internal(tables, tcx).into(),
                        projections.internal(tables, tcx).into(),
                    )),
                    variance.internal(tables, tcx).into(),
                )
            }
            StatementKind::Coverage(_coverage_kind) => {
                unimplemented!("cannot convert back from an opaque field")
            }
            StatementKind::Intrinsic(non_diverging_intrinsic) => {
                rustc_middle::mir::StatementKind::Intrinsic(
                    non_diverging_intrinsic.internal(tables, tcx).into(),
                )
            }
            StatementKind::ConstEvalCounter => rustc_middle::mir::StatementKind::ConstEvalCounter,
            StatementKind::Nop => rustc_middle::mir::StatementKind::Nop,
        }
    }
}

impl RustcInternal for Statement {
    type T<'tcx> = rustc_middle::mir::Statement<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_middle::mir::Statement {
            source_info: rustc_middle::mir::SourceInfo::outermost(
                // TODO: is this a good default value?
                self.span.internal(tables, tcx).into(),
            ),
            kind: self.kind.internal(tables, tcx).into(),
        }
    }
}

impl RustcInternal for UnwindAction {
    type T<'tcx> = rustc_middle::mir::UnwindAction;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            UnwindAction::Continue => rustc_middle::mir::UnwindAction::Continue,
            UnwindAction::Unreachable => rustc_middle::mir::UnwindAction::Unreachable,
            UnwindAction::Terminate => rustc_middle::mir::UnwindAction::Terminate(
                rustc_middle::mir::UnwindTerminateReason::Abi, // TODO: is this a good default value?
            ),
            UnwindAction::Cleanup(basic_block_idx) => rustc_middle::mir::UnwindAction::Cleanup(
                rustc_middle::mir::BasicBlock::from_usize(*basic_block_idx),
            ),
        }
    }
}

impl RustcInternal for SwitchTargets {
    type T<'tcx> = rustc_middle::mir::SwitchTargets;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_middle::mir::SwitchTargets::new(
            self.branches().map(|(value, basic_block_idx)| {
                (value, rustc_middle::mir::BasicBlock::from_usize(basic_block_idx))
            }),
            rustc_middle::mir::BasicBlock::from_usize(self.otherwise()),
        )
    }
}

impl RustcInternal for CoroutineDesugaring {
    type T<'tcx> = rustc_hir::CoroutineDesugaring;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            CoroutineDesugaring::Async => rustc_hir::CoroutineDesugaring::Async,
            CoroutineDesugaring::Gen => rustc_hir::CoroutineDesugaring::Gen,
            CoroutineDesugaring::AsyncGen => rustc_hir::CoroutineDesugaring::AsyncGen,
        }
    }
}

impl RustcInternal for CoroutineSource {
    type T<'tcx> = rustc_hir::CoroutineSource;

    fn internal<'tcx>(&self, _tables: &mut Tables<'_>, _tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            CoroutineSource::Block => rustc_hir::CoroutineSource::Block,
            CoroutineSource::Closure => rustc_hir::CoroutineSource::Closure,
            CoroutineSource::Fn => rustc_hir::CoroutineSource::Fn,
        }
    }
}

impl RustcInternal for CoroutineKind {
    type T<'tcx> = rustc_hir::CoroutineKind;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            CoroutineKind::Desugared(coroutine_desugaring, coroutine_source) => {
                rustc_hir::CoroutineKind::Desugared(
                    coroutine_desugaring.internal(tables, tcx).into(),
                    coroutine_source.internal(tables, tcx).into(),
                )
            }
            CoroutineKind::Coroutine(movability) => {
                rustc_hir::CoroutineKind::Coroutine(movability.internal(tables, tcx).into())
            }
        }
    }
}

impl RustcInternal for AssertMessage {
    type T<'tcx> = rustc_middle::mir::AssertMessage<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            AssertMessage::BoundsCheck { len, index } => {
                rustc_middle::mir::AssertMessage::BoundsCheck {
                    len: len.internal(tables, tcx).into(),
                    index: index.internal(tables, tcx).into(),
                }
            }
            AssertMessage::Overflow(bin_op, left_operand, right_operand) => {
                rustc_middle::mir::AssertMessage::Overflow(
                    bin_op.internal(tables, tcx).into(),
                    left_operand.internal(tables, tcx).into(),
                    right_operand.internal(tables, tcx).into(),
                )
            }
            AssertMessage::OverflowNeg(operand) => {
                rustc_middle::mir::AssertMessage::OverflowNeg(operand.internal(tables, tcx).into())
            }
            AssertMessage::DivisionByZero(operand) => {
                rustc_middle::mir::AssertMessage::DivisionByZero(
                    operand.internal(tables, tcx).into(),
                )
            }
            AssertMessage::RemainderByZero(operand) => {
                rustc_middle::mir::AssertMessage::RemainderByZero(
                    operand.internal(tables, tcx).into(),
                )
            }
            AssertMessage::ResumedAfterReturn(coroutine_kind) => {
                rustc_middle::mir::AssertMessage::ResumedAfterReturn(
                    coroutine_kind.internal(tables, tcx).into(),
                )
            }
            AssertMessage::ResumedAfterPanic(coroutine_kind) => {
                rustc_middle::mir::AssertMessage::ResumedAfterPanic(
                    coroutine_kind.internal(tables, tcx).into(),
                )
            }
            AssertMessage::MisalignedPointerDereference { required, found } => {
                rustc_middle::mir::AssertMessage::MisalignedPointerDereference {
                    required: required.internal(tables, tcx).into(),
                    found: found.internal(tables, tcx).into(),
                }
            }
        }
    }
}

impl RustcInternal for TerminatorKind {
    type T<'tcx> = rustc_middle::mir::TerminatorKind<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        match self {
            TerminatorKind::Goto { target } => rustc_middle::mir::TerminatorKind::Goto {
                target: rustc_middle::mir::BasicBlock::from_usize(*target),
            },
            TerminatorKind::SwitchInt { discr, targets } => {
                rustc_middle::mir::TerminatorKind::SwitchInt {
                    discr: discr.internal(tables, tcx).into(),
                    targets: targets.internal(tables, tcx).into(),
                }
            }
            TerminatorKind::Resume => rustc_middle::mir::TerminatorKind::UnwindResume,
            TerminatorKind::Abort => rustc_middle::mir::TerminatorKind::UnwindTerminate(
                rustc_middle::mir::UnwindTerminateReason::Abi,
            ),
            TerminatorKind::Return => rustc_middle::mir::TerminatorKind::Return,
            TerminatorKind::Unreachable => rustc_middle::mir::TerminatorKind::Unreachable,
            TerminatorKind::Drop { place, target, unwind } => {
                rustc_middle::mir::TerminatorKind::Drop {
                    place: place.internal(tables, tcx).into(),
                    target: rustc_middle::mir::BasicBlock::from_usize(*target),
                    unwind: unwind.internal(tables, tcx).into(),
                    replace: false, // TODO: is this a good default value?
                }
            }
            TerminatorKind::Call { func, args, destination, target, unwind } => {
                rustc_middle::mir::TerminatorKind::Call {
                    func: func.internal(tables, tcx).into(),
                    args: Box::from_iter(args.iter().map(|arg| {
                        rustc_span::source_map::dummy_spanned(arg.internal(tables, tcx).into())
                    })),
                    destination: destination.internal(tables, tcx).into(),
                    target: target.map(|basic_block_idx| {
                        rustc_middle::mir::BasicBlock::from_usize(basic_block_idx)
                    }),
                    unwind: unwind.internal(tables, tcx).into(),
                    call_source: rustc_middle::mir::CallSource::Misc, // TODO: is this a good default value?
                    fn_span: rustc_span::DUMMY_SP, // TODO: is this a good default value?
                }
            }
            TerminatorKind::Assert { cond, expected, msg, target, unwind } => {
                rustc_middle::mir::TerminatorKind::Assert {
                    cond: cond.internal(tables, tcx).into(),
                    expected: *expected,
                    msg: Box::new(msg.internal(tables, tcx).into()),
                    target: rustc_middle::mir::BasicBlock::from_usize(*target),
                    unwind: unwind.internal(tables, tcx).into(),
                }
            }
            TerminatorKind::InlineAsm { .. } => todo!(),
        }
    }
}

impl RustcInternal for Terminator {
    type T<'tcx> = rustc_middle::mir::Terminator<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_middle::mir::Terminator {
            source_info: rustc_middle::mir::SourceInfo::outermost(
                self.span.internal(tables, tcx).into(),
            ),
            kind: self.kind.internal(tables, tcx).into(),
        }
    }
}

impl RustcInternal for LocalDecl {
    type T<'tcx> = rustc_middle::mir::LocalDecl<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        rustc_middle::mir::LocalDecl {
            mutability: self.mutability.internal(tables, tcx).into(),
            local_info: rustc_middle::mir::ClearCrossCrate::Set(Box::new(
                rustc_middle::mir::LocalInfo::Boring,
            )),
            ty: self.ty.internal(tables, tcx).into(),
            user_ty: None, // TODO: is this a good default value?
            source_info: rustc_middle::mir::SourceInfo::outermost(
                self.span.internal(tables, tcx).into(),
            ),
        }
    }
}

impl RustcInternal for Body {
    type T<'tcx> = rustc_middle::mir::Body<'tcx>;

    fn internal<'tcx>(&self, tables: &mut Tables<'_>, tcx: TyCtxt<'tcx>) -> Self::T<'tcx> {
        let internal_basic_blocks = rustc_index::IndexVec::from_raw(
            self.blocks
                .iter()
                .map(|stable_basic_block| rustc_middle::mir::BasicBlockData {
                    statements: stable_basic_block
                        .statements
                        .iter()
                        .map(|statement| statement.internal(tables, tcx).into())
                        .collect(),
                    terminator: Some(stable_basic_block.terminator.internal(tables, tcx).into()),
                    is_cleanup: false,
                })
                .collect(),
        );
        let local_decls = rustc_index::IndexVec::from_raw(
            self.locals()
                .iter()
                .map(|local_decl| local_decl.internal(tables, tcx).into())
                .collect(),
        );
        // TODO: this is lossy, I wonder how should we signal this to the user.
        rustc_middle::mir::Body::new(
            rustc_middle::mir::MirSource::item(rustc_hir::def_id::CRATE_DEF_ID.to_def_id()),
            internal_basic_blocks,
            rustc_index::IndexVec::new(),
            local_decls,
            rustc_index::IndexVec::new(),
            self.arg_count(),
            Vec::new(),
            rustc_span::DUMMY_SP,
            None,
            None,
        )
    }
}
