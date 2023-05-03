use crate::ty::{self, AliasTy, PolyTraitRef, Term, Ty, TyCtxt};
use rustc_hir::def_id::DefId;

pub type PolyProjectionPredicate<'tcx> = ty::Binder<'tcx, ProjectionPredicate<'tcx>>;

/// This kind of predicate has no *direct* correspondent in the
/// syntax, but it roughly corresponds to the syntactic forms:
///
/// 1. `T: TraitRef<..., Item = Type>`
/// 2. `<T as TraitRef<...>>::Item == Type` (NYI)
///
/// In particular, form #1 is "desugared" to the combination of a
/// normal trait predicate (`T: TraitRef<...>`) and one of these
/// predicates. Form #2 is a broader form in that it also permits
/// equality between arbitrary types. Processing an instance of
/// Form #2 eventually yields one of these `ProjectionPredicate`
/// instances to normalize the LHS.
#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct ProjectionPredicate<'tcx> {
    pub projection_ty: AliasTy<'tcx>,
    pub term: Term<'tcx>,
}

impl<'tcx> ProjectionPredicate<'tcx> {
    pub fn self_ty(self) -> Ty<'tcx> {
        self.projection_ty.self_ty()
    }

    pub fn with_self_ty(self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> ProjectionPredicate<'tcx> {
        Self { projection_ty: self.projection_ty.with_self_ty(tcx, self_ty), ..self }
    }

    pub fn trait_def_id(self, tcx: TyCtxt<'tcx>) -> DefId {
        self.projection_ty.trait_def_id(tcx)
    }

    pub fn def_id(self) -> DefId {
        self.projection_ty.def_id
    }
}

impl<'tcx> PolyProjectionPredicate<'tcx> {
    /// Returns the `DefId` of the trait of the associated item being projected.
    #[inline]
    pub fn trait_def_id(&self, tcx: TyCtxt<'tcx>) -> DefId {
        self.skip_binder().projection_ty.trait_def_id(tcx)
    }

    /// Get the [PolyTraitRef] required for this projection to be well formed.
    /// Note that for generic associated types the predicates of the associated
    /// type also need to be checked.
    #[inline]
    pub fn required_poly_trait_ref(&self, tcx: TyCtxt<'tcx>) -> PolyTraitRef<'tcx> {
        // Note: unlike with `TraitRef::to_poly_trait_ref()`,
        // `self.0.trait_ref` is permitted to have escaping regions.
        // This is because here `self` has a `Binder` and so does our
        // return value, so we are preserving the number of binding
        // levels.
        self.map_bound(|predicate| predicate.projection_ty.trait_ref(tcx))
    }

    pub fn term(&self) -> ty::Binder<'tcx, Term<'tcx>> {
        self.map_bound(|predicate| predicate.term)
    }

    /// The `DefId` of the `TraitItem` for the associated type.
    ///
    /// Note that this is not the `DefId` of the `TraitRef` containing this
    /// associated type, which is in `tcx.associated_item(projection_def_id()).container`.
    pub fn projection_def_id(&self) -> DefId {
        // Ok to skip binder since trait `DefId` does not care about regions.
        self.skip_binder().projection_ty.def_id
    }
}
