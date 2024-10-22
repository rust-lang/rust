//! `TypeFoldable` implementations for MIR types
use rustc_index::bit_set::BitMatrix;

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
    BitMatrix<CoroutineSavedLocal, CoroutineSavedLocal>,
}

TrivialTypeTraversalImpls! {
    ConstValue<'tcx>,
    NullOp<'tcx>,
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for &'tcx ty::List<PlaceElem<'tcx>> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        ty::util::fold_list(self, folder, |tcx, v| tcx.mk_place_elems(v))
    }
}
