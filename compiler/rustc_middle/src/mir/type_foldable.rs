//! `TypeFoldable` implementations for MIR types

use super::*;
use crate::ty;

CloneLiftImpls! {
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
    NullOp,
    hir::Movability,
    BasicBlock,
    SwitchTargets,
    GeneratorKind,
    GeneratorSavedLocal,
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for &'tcx ty::List<PlaceElem<'tcx>> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        ty::util::fold_list(self, folder, |tcx, v| tcx.mk_place_elems(v))
    }
}
