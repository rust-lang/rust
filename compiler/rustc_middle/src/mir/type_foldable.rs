//! `TypeFoldable` implementations for MIR types

use rustc_ast::InlineAsmTemplatePiece;

use super::*;
use crate::ty;

TrivialTypeTraversalAndLiftImpls! {
    BlockTailInfo,
    MirPhase,
    SourceInfo,
    FakeReadCause,
    RetagKind,
    SourceScope,
    SourceScopeLocalData,
    UserTypeAnnotationIndex,
    BorrowKind,
    CastKind,
    BinOp,
    NullOp,
    UnOp,
    hir::Movability,
    BasicBlock,
    SwitchTargets,
    GeneratorKind,
    GeneratorSavedLocal,
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx [InlineAsmTemplatePiece] {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, _folder: &mut F) -> Result<Self, F::Error> {
        Ok(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx [Span] {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, _folder: &mut F) -> Result<Self, F::Error> {
        Ok(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::List<PlaceElem<'tcx>> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        ty::util::fold_list(self, folder, |tcx, v| tcx.intern_place_elems(v))
    }
}

impl<'tcx, R: Idx, C: Idx> TypeFoldable<'tcx> for BitMatrix<R, C> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, _: &mut F) -> Result<Self, F::Error> {
        Ok(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ConstantKind<'tcx> {
    #[inline(always)]
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_mir_const(self)
    }
}

impl<'tcx> TypeSuperFoldable<'tcx> for ConstantKind<'tcx> {
    fn try_super_fold_with<F: FallibleTypeFolder<'tcx>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        match self {
            ConstantKind::Ty(c) => Ok(ConstantKind::Ty(c.try_fold_with(folder)?)),
            ConstantKind::Val(v, t) => Ok(ConstantKind::Val(v, t.try_fold_with(folder)?)),
            ConstantKind::Unevaluated(uv, t) => {
                Ok(ConstantKind::Unevaluated(uv.try_fold_with(folder)?, t.try_fold_with(folder)?))
            }
        }
    }
}
