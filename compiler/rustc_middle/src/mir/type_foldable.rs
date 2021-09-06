//! `TypeFoldable` implementations for MIR types

use super::*;
use crate::ty;
use rustc_data_structures::functor::IdFunctor;

TrivialTypeFoldableAndLiftImpls! {
    BlockTailInfo,
    MirPhase,
    SourceInfo,
    FakeReadCause,
    RetagKind,
    SourceScope,
    SourceScopeLocalData,
    UserTypeAnnotationIndex,
}

impl<'tcx> TypeFoldable<'tcx> for Terminator<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        use crate::mir::TerminatorKind::*;

        let kind = match self.kind {
            Goto { target } => Goto { target },
            SwitchInt { discr, switch_ty, targets } => SwitchInt {
                discr: discr.fold_with(folder),
                switch_ty: switch_ty.fold_with(folder),
                targets,
            },
            Drop { place, target, unwind } => {
                Drop { place: place.fold_with(folder), target, unwind }
            }
            DropAndReplace { place, value, target, unwind } => DropAndReplace {
                place: place.fold_with(folder),
                value: value.fold_with(folder),
                target,
                unwind,
            },
            Yield { value, resume, resume_arg, drop } => Yield {
                value: value.fold_with(folder),
                resume,
                resume_arg: resume_arg.fold_with(folder),
                drop,
            },
            Call { func, args, destination, cleanup, from_hir_call, fn_span } => {
                let dest = destination.map(|(loc, dest)| (loc.fold_with(folder), dest));

                Call {
                    func: func.fold_with(folder),
                    args: args.fold_with(folder),
                    destination: dest,
                    cleanup,
                    from_hir_call,
                    fn_span,
                }
            }
            Assert { cond, expected, msg, target, cleanup } => {
                use AssertKind::*;
                let msg = match msg {
                    BoundsCheck { len, index } => {
                        BoundsCheck { len: len.fold_with(folder), index: index.fold_with(folder) }
                    }
                    Overflow(op, l, r) => Overflow(op, l.fold_with(folder), r.fold_with(folder)),
                    OverflowNeg(op) => OverflowNeg(op.fold_with(folder)),
                    DivisionByZero(op) => DivisionByZero(op.fold_with(folder)),
                    RemainderByZero(op) => RemainderByZero(op.fold_with(folder)),
                    ResumedAfterReturn(_) | ResumedAfterPanic(_) => msg,
                };
                Assert { cond: cond.fold_with(folder), expected, msg, target, cleanup }
            }
            GeneratorDrop => GeneratorDrop,
            Resume => Resume,
            Abort => Abort,
            Return => Return,
            Unreachable => Unreachable,
            FalseEdge { real_target, imaginary_target } => {
                FalseEdge { real_target, imaginary_target }
            }
            FalseUnwind { real_target, unwind } => FalseUnwind { real_target, unwind },
            InlineAsm { template, operands, options, line_spans, destination } => InlineAsm {
                template,
                operands: operands.fold_with(folder),
                options,
                line_spans,
                destination,
            },
        };
        Terminator { source_info: self.source_info, kind }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        use crate::mir::TerminatorKind::*;

        match self.kind {
            SwitchInt { ref discr, switch_ty, .. } => {
                discr.visit_with(visitor)?;
                switch_ty.visit_with(visitor)
            }
            Drop { ref place, .. } => place.visit_with(visitor),
            DropAndReplace { ref place, ref value, .. } => {
                place.visit_with(visitor)?;
                value.visit_with(visitor)
            }
            Yield { ref value, .. } => value.visit_with(visitor),
            Call { ref func, ref args, ref destination, .. } => {
                if let Some((ref loc, _)) = *destination {
                    loc.visit_with(visitor)?;
                };
                func.visit_with(visitor)?;
                args.visit_with(visitor)
            }
            Assert { ref cond, ref msg, .. } => {
                cond.visit_with(visitor)?;
                use AssertKind::*;
                match msg {
                    BoundsCheck { ref len, ref index } => {
                        len.visit_with(visitor)?;
                        index.visit_with(visitor)
                    }
                    Overflow(_, l, r) => {
                        l.visit_with(visitor)?;
                        r.visit_with(visitor)
                    }
                    OverflowNeg(op) | DivisionByZero(op) | RemainderByZero(op) => {
                        op.visit_with(visitor)
                    }
                    ResumedAfterReturn(_) | ResumedAfterPanic(_) => ControlFlow::CONTINUE,
                }
            }
            InlineAsm { ref operands, .. } => operands.visit_with(visitor),
            Goto { .. }
            | Resume
            | Abort
            | Return
            | GeneratorDrop
            | Unreachable
            | FalseEdge { .. }
            | FalseUnwind { .. } => ControlFlow::CONTINUE,
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for GeneratorKind {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, _: &mut F) -> Self {
        self
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }
}

impl<'tcx> TypeFoldable<'tcx> for Place<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        Place { local: self.local.fold_with(folder), projection: self.projection.fold_with(folder) }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.local.visit_with(visitor)?;
        self.projection.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<PlaceElem<'tcx>> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        ty::util::fold_list(self, folder, |tcx, v| tcx.intern_place_elems(v))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for Rvalue<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        use crate::mir::Rvalue::*;
        match self {
            Use(op) => Use(op.fold_with(folder)),
            Repeat(op, len) => Repeat(op.fold_with(folder), len.fold_with(folder)),
            ThreadLocalRef(did) => ThreadLocalRef(did.fold_with(folder)),
            Ref(region, bk, place) => Ref(region.fold_with(folder), bk, place.fold_with(folder)),
            AddressOf(mutability, place) => AddressOf(mutability, place.fold_with(folder)),
            Len(place) => Len(place.fold_with(folder)),
            Cast(kind, op, ty) => Cast(kind, op.fold_with(folder), ty.fold_with(folder)),
            BinaryOp(op, box (rhs, lhs)) => {
                BinaryOp(op, Box::new((rhs.fold_with(folder), lhs.fold_with(folder))))
            }
            CheckedBinaryOp(op, box (rhs, lhs)) => {
                CheckedBinaryOp(op, Box::new((rhs.fold_with(folder), lhs.fold_with(folder))))
            }
            UnaryOp(op, val) => UnaryOp(op, val.fold_with(folder)),
            Discriminant(place) => Discriminant(place.fold_with(folder)),
            NullaryOp(op, ty) => NullaryOp(op, ty.fold_with(folder)),
            Aggregate(kind, fields) => {
                let kind = kind.map_id(|kind| match kind {
                    AggregateKind::Array(ty) => AggregateKind::Array(ty.fold_with(folder)),
                    AggregateKind::Tuple => AggregateKind::Tuple,
                    AggregateKind::Adt(def, v, substs, user_ty, n) => AggregateKind::Adt(
                        def,
                        v,
                        substs.fold_with(folder),
                        user_ty.fold_with(folder),
                        n,
                    ),
                    AggregateKind::Closure(id, substs) => {
                        AggregateKind::Closure(id, substs.fold_with(folder))
                    }
                    AggregateKind::Generator(id, substs, movablity) => {
                        AggregateKind::Generator(id, substs.fold_with(folder), movablity)
                    }
                });
                Aggregate(kind, fields.fold_with(folder))
            }
            ShallowInitBox(op, ty) => ShallowInitBox(op.fold_with(folder), ty.fold_with(folder)),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        use crate::mir::Rvalue::*;
        match *self {
            Use(ref op) => op.visit_with(visitor),
            Repeat(ref op, _) => op.visit_with(visitor),
            ThreadLocalRef(did) => did.visit_with(visitor),
            Ref(region, _, ref place) => {
                region.visit_with(visitor)?;
                place.visit_with(visitor)
            }
            AddressOf(_, ref place) => place.visit_with(visitor),
            Len(ref place) => place.visit_with(visitor),
            Cast(_, ref op, ty) => {
                op.visit_with(visitor)?;
                ty.visit_with(visitor)
            }
            BinaryOp(_, box (ref rhs, ref lhs)) | CheckedBinaryOp(_, box (ref rhs, ref lhs)) => {
                rhs.visit_with(visitor)?;
                lhs.visit_with(visitor)
            }
            UnaryOp(_, ref val) => val.visit_with(visitor),
            Discriminant(ref place) => place.visit_with(visitor),
            NullaryOp(_, ty) => ty.visit_with(visitor),
            Aggregate(ref kind, ref fields) => {
                match **kind {
                    AggregateKind::Array(ty) => {
                        ty.visit_with(visitor)?;
                    }
                    AggregateKind::Tuple => {}
                    AggregateKind::Adt(_, _, substs, user_ty, _) => {
                        substs.visit_with(visitor)?;
                        user_ty.visit_with(visitor)?;
                    }
                    AggregateKind::Closure(_, substs) => {
                        substs.visit_with(visitor)?;
                    }
                    AggregateKind::Generator(_, substs, _) => {
                        substs.visit_with(visitor)?;
                    }
                }
                fields.visit_with(visitor)
            }
            ShallowInitBox(ref op, ty) => {
                op.visit_with(visitor)?;
                ty.visit_with(visitor)
            }
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for Operand<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        match self {
            Operand::Copy(place) => Operand::Copy(place.fold_with(folder)),
            Operand::Move(place) => Operand::Move(place.fold_with(folder)),
            Operand::Constant(c) => Operand::Constant(c.fold_with(folder)),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match *self {
            Operand::Copy(ref place) | Operand::Move(ref place) => place.visit_with(visitor),
            Operand::Constant(ref c) => c.visit_with(visitor),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for PlaceElem<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        use crate::mir::ProjectionElem::*;

        match self {
            Deref => Deref,
            Field(f, ty) => Field(f, ty.fold_with(folder)),
            Index(v) => Index(v.fold_with(folder)),
            Downcast(symbol, variantidx) => Downcast(symbol, variantidx),
            ConstantIndex { offset, min_length, from_end } => {
                ConstantIndex { offset, min_length, from_end }
            }
            Subslice { from, to, from_end } => Subslice { from, to, from_end },
        }
    }

    fn super_visit_with<Vs: TypeVisitor<'tcx>>(
        &self,
        visitor: &mut Vs,
    ) -> ControlFlow<Vs::BreakTy> {
        use crate::mir::ProjectionElem::*;

        match self {
            Field(_, ty) => ty.visit_with(visitor),
            Index(v) => v.visit_with(visitor),
            _ => ControlFlow::CONTINUE,
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for Field {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, _: &mut F) -> Self {
        self
    }
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }
}

impl<'tcx> TypeFoldable<'tcx> for GeneratorSavedLocal {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, _: &mut F) -> Self {
        self
    }
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }
}

impl<'tcx, R: Idx, C: Idx> TypeFoldable<'tcx> for BitMatrix<R, C> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, _: &mut F) -> Self {
        self
    }
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }
}

impl<'tcx> TypeFoldable<'tcx> for Constant<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        Constant {
            span: self.span,
            user_ty: self.user_ty.fold_with(folder),
            literal: self.literal.fold_with(folder),
        }
    }
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.literal.visit_with(visitor)?;
        self.user_ty.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ConstantKind<'tcx> {
    #[inline(always)]
    fn fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        folder.fold_mir_const(self)
    }

    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        match self {
            ConstantKind::Ty(c) => ConstantKind::Ty(c.fold_with(folder)),
            ConstantKind::Val(v, t) => ConstantKind::Val(v, t.fold_with(folder)),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match *self {
            ConstantKind::Ty(c) => c.visit_with(visitor),
            ConstantKind::Val(_, t) => t.visit_with(visitor),
        }
    }
}
