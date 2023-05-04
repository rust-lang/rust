use crate::ty::Ty;

#[derive(Clone, Debug, PartialEq, Eq, Copy, Hash, TyEncodable, TyDecodable, HashStable)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct ClosureSizeProfileData<'tcx> {
    /// Tuple containing the types of closure captures before the feature `capture_disjoint_fields`
    pub before_feature_tys: Ty<'tcx>,
    /// Tuple containing the types of closure captures after the feature `capture_disjoint_fields`
    pub after_feature_tys: Ty<'tcx>,
}
