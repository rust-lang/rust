//! `TypeVisitable` implementations for MIR types

use super::*;
use crate::ty;

impl<'tcx> TypeVisitable<'tcx> for Terminator<'tcx> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
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
                destination.visit_with(visitor)?;
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

impl<'tcx> TypeVisitable<'tcx> for GeneratorKind {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, _: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }
}

impl<'tcx> TypeVisitable<'tcx> for Place<'tcx> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.local.visit_with(visitor)?;
        self.projection.visit_with(visitor)
    }
}

impl<'tcx> TypeVisitable<'tcx> for &'tcx ty::List<PlaceElem<'tcx>> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<'tcx> TypeVisitable<'tcx> for Rvalue<'tcx> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
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

impl<'tcx> TypeVisitable<'tcx> for Operand<'tcx> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match *self {
            Operand::Copy(ref place) | Operand::Move(ref place) => place.visit_with(visitor),
            Operand::Constant(ref c) => c.visit_with(visitor),
        }
    }
}

impl<'tcx> TypeVisitable<'tcx> for PlaceElem<'tcx> {
    fn visit_with<Vs: TypeVisitor<'tcx>>(&self, visitor: &mut Vs) -> ControlFlow<Vs::BreakTy> {
        use crate::mir::ProjectionElem::*;

        match self {
            Field(_, ty) => ty.visit_with(visitor),
            Index(v) => v.visit_with(visitor),
            _ => ControlFlow::CONTINUE,
        }
    }
}

impl<'tcx> TypeVisitable<'tcx> for Field {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, _: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }
}

impl<'tcx> TypeVisitable<'tcx> for GeneratorSavedLocal {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, _: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }
}

impl<'tcx, R: Idx, C: Idx> TypeVisitable<'tcx> for BitMatrix<R, C> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, _: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }
}

impl<'tcx> TypeVisitable<'tcx> for Constant<'tcx> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.literal.visit_with(visitor)?;
        self.user_ty.visit_with(visitor)
    }
}

impl<'tcx> TypeVisitable<'tcx> for ConstantKind<'tcx> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        visitor.visit_mir_const(*self)
    }
}

impl<'tcx> TypeSuperVisitable<'tcx> for ConstantKind<'tcx> {
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match *self {
            ConstantKind::Ty(c) => c.visit_with(visitor),
            ConstantKind::Val(_, t) => t.visit_with(visitor),
        }
    }
}
