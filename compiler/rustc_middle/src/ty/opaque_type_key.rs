use rustc_hir::def_id::LocalDefId;

use crate::ty::SubstsRef;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable, TyEncodable, TyDecodable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct OpaqueTypeKey<'tcx> {
    pub def_id: LocalDefId,
    pub substs: SubstsRef<'tcx>,
}
