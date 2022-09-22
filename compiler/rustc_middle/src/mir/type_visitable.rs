//! `TypeVisitable` implementations for MIR types

use super::*;

impl<'tcx, R: Idx, C: Idx> TypeVisitable<'tcx> for BitMatrix<R, C> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, _: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
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
            ConstantKind::Unevaluated(uv, t) => {
                uv.visit_with(visitor)?;
                t.visit_with(visitor)
            }
        }
    }
}
