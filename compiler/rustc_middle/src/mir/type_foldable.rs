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

TrivialTypeTraversalImpls! {
    for <'tcx> {
        ConstValue<'tcx>,
    }
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
