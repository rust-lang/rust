//! `TypeFoldable` implementations for MIR types

use rustc_ast::InlineAsmTemplatePiece;
use rustc_hir::def_id::LocalDefId;

use super::*;

TrivialTypeTraversalImpls! {
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
    BasicBlock,
    SwitchTargets,
    CoroutineKind,
    CoroutineSavedLocal,
}

TrivialTypeTraversalImpls! {
    ConstValue<'tcx>,
    NullOp<'tcx>,
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for &'tcx [InlineAsmTemplatePiece] {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        _folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(self)
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for &'tcx [Span] {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        _folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(self)
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for &'tcx ty::List<LocalDefId> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        _folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(self)
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for &'tcx ty::List<PlaceElem<'tcx>> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        ty::util::fold_list(self, folder, |tcx, v| tcx.mk_place_elems(v))
    }
}
