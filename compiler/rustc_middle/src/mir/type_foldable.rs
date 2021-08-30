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
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        use crate::mir::TerminatorKind::*;

        let kind = match self.kind {
            Goto { target } => Goto { target },
            SwitchInt { discr, switch_ty, targets } => SwitchInt {
                discr: discr.try_fold_with(folder)?,
                switch_ty: switch_ty.try_fold_with(folder)?,
                targets,
            },
            Drop { place, target, unwind } => {
                Drop { place: place.try_fold_with(folder)?, target, unwind }
            }
            DropAndReplace { place, value, target, unwind } => DropAndReplace {
                place: place.try_fold_with(folder)?,
                value: value.try_fold_with(folder)?,
                target,
                unwind,
            },
            Yield { value, resume, resume_arg, drop } => Yield {
                value: value.try_fold_with(folder)?,
                resume,
                resume_arg: resume_arg.try_fold_with(folder)?,
                drop,
            },
            Call { func, args, destination, cleanup, from_hir_call, fn_span } => {
                let dest = destination
                    .map(|(loc, dest)| (loc.try_fold_with(folder).map(|loc| (loc, dest))))
                    .transpose()?;

                Call {
                    func: func.try_fold_with(folder)?,
                    args: args.try_fold_with(folder)?,
                    destination: dest,
                    cleanup,
                    from_hir_call,
                    fn_span,
                }
            }
            Assert { cond, expected, msg, target, cleanup } => {
                use AssertKind::*;
                let msg = match msg {
                    BoundsCheck { len, index } => BoundsCheck {
                        len: len.try_fold_with(folder)?,
                        index: index.try_fold_with(folder)?,
                    },
                    Overflow(op, l, r) => {
                        Overflow(op, l.try_fold_with(folder)?, r.try_fold_with(folder)?)
                    }
                    OverflowNeg(op) => OverflowNeg(op.try_fold_with(folder)?),
                    DivisionByZero(op) => DivisionByZero(op.try_fold_with(folder)?),
                    RemainderByZero(op) => RemainderByZero(op.try_fold_with(folder)?),
                    ResumedAfterReturn(_) | ResumedAfterPanic(_) => msg,
                };
                Assert { cond: cond.try_fold_with(folder)?, expected, msg, target, cleanup }
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
            InlineAsm { template, operands, options, line_spans, destination, cleanup } => {
                InlineAsm {
                    template,
                    operands: operands.try_fold_with(folder)?,
                    options,
                    line_spans,
                    destination,
                    cleanup,
                }
            }
        };
        Ok(Terminator { source_info: self.source_info, kind })
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
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(self, _: &mut F) -> Result<Self, F::Error> {
        Ok(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }
}

impl<'tcx> TypeFoldable<'tcx> for Place<'tcx> {
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(Place {
            local: self.local.try_fold_with(folder)?,
            projection: self.projection.try_fold_with(folder)?,
        })
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.local.visit_with(visitor)?;
        self.projection.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<PlaceElem<'tcx>> {
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        ty::util::fold_list(self, folder, |tcx, v| tcx.intern_place_elems(v))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for Rvalue<'tcx> {
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        use crate::mir::Rvalue::*;
        Ok(match self {
            Use(op) => Use(op.try_fold_with(folder)?),
            Repeat(op, len) => Repeat(op.try_fold_with(folder)?, len.try_fold_with(folder)?),
            ThreadLocalRef(did) => ThreadLocalRef(did.try_fold_with(folder)?),
            Ref(region, bk, place) => {
                Ref(region.try_fold_with(folder)?, bk, place.try_fold_with(folder)?)
            }
            AddressOf(mutability, place) => AddressOf(mutability, place.try_fold_with(folder)?),
            Len(place) => Len(place.try_fold_with(folder)?),
            Cast(kind, op, ty) => Cast(kind, op.try_fold_with(folder)?, ty.try_fold_with(folder)?),
            BinaryOp(op, box (rhs, lhs)) => {
                BinaryOp(op, Box::new((rhs.try_fold_with(folder)?, lhs.try_fold_with(folder)?)))
            }
            CheckedBinaryOp(op, box (rhs, lhs)) => CheckedBinaryOp(
                op,
                Box::new((rhs.try_fold_with(folder)?, lhs.try_fold_with(folder)?)),
            ),
            UnaryOp(op, val) => UnaryOp(op, val.try_fold_with(folder)?),
            Discriminant(place) => Discriminant(place.try_fold_with(folder)?),
            NullaryOp(op, ty) => NullaryOp(op, ty.try_fold_with(folder)?),
            Aggregate(kind, fields) => {
                let kind = kind.try_map_id(|kind| {
                    Ok(match kind {
                        AggregateKind::Array(ty) => AggregateKind::Array(ty.try_fold_with(folder)?),
                        AggregateKind::Tuple => AggregateKind::Tuple,
                        AggregateKind::Adt(def, v, substs, user_ty, n) => AggregateKind::Adt(
                            def,
                            v,
                            substs.try_fold_with(folder)?,
                            user_ty.try_fold_with(folder)?,
                            n,
                        ),
                        AggregateKind::Closure(id, substs) => {
                            AggregateKind::Closure(id, substs.try_fold_with(folder)?)
                        }
                        AggregateKind::Generator(id, substs, movablity) => {
                            AggregateKind::Generator(id, substs.try_fold_with(folder)?, movablity)
                        }
                    })
                })?;
                Aggregate(kind, fields.try_fold_with(folder)?)
            }
            ShallowInitBox(op, ty) => {
                ShallowInitBox(op.try_fold_with(folder)?, ty.try_fold_with(folder)?)
            }
        })
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
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(match self {
            Operand::Copy(place) => Operand::Copy(place.try_fold_with(folder)?),
            Operand::Move(place) => Operand::Move(place.try_fold_with(folder)?),
            Operand::Constant(c) => Operand::Constant(c.try_fold_with(folder)?),
        })
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match *self {
            Operand::Copy(ref place) | Operand::Move(ref place) => place.visit_with(visitor),
            Operand::Constant(ref c) => c.visit_with(visitor),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for PlaceElem<'tcx> {
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        use crate::mir::ProjectionElem::*;

        Ok(match self {
            Deref => Deref,
            Field(f, ty) => Field(f, ty.try_fold_with(folder)?),
            Index(v) => Index(v.try_fold_with(folder)?),
            Downcast(symbol, variantidx) => Downcast(symbol, variantidx),
            ConstantIndex { offset, min_length, from_end } => {
                ConstantIndex { offset, min_length, from_end }
            }
            Subslice { from, to, from_end } => Subslice { from, to, from_end },
        })
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
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(self, _: &mut F) -> Result<Self, F::Error> {
        Ok(self)
    }
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }
}

impl<'tcx> TypeFoldable<'tcx> for GeneratorSavedLocal {
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(self, _: &mut F) -> Result<Self, F::Error> {
        Ok(self)
    }
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }
}

impl<'tcx, R: Idx, C: Idx> TypeFoldable<'tcx> for BitMatrix<R, C> {
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(self, _: &mut F) -> Result<Self, F::Error> {
        Ok(self)
    }
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }
}

impl<'tcx> TypeFoldable<'tcx> for Constant<'tcx> {
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(Constant {
            span: self.span,
            user_ty: self.user_ty.try_fold_with(folder)?,
            literal: self.literal.try_fold_with(folder)?,
        })
    }
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.literal.visit_with(visitor)?;
        self.user_ty.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ConstantKind<'tcx> {
    #[inline(always)]
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_mir_const(self)
    }

    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        match self {
            ConstantKind::Ty(c) => Ok(ConstantKind::Ty(c.try_fold_with(folder)?)),
            ConstantKind::Val(v, t) => Ok(ConstantKind::Val(v, t.try_fold_with(folder)?)),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match *self {
            ConstantKind::Ty(c) => c.visit_with(visitor),
            ConstantKind::Val(_, t) => t.visit_with(visitor),
        }
    }
}
