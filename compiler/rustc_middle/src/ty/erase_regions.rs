use tracing::debug;

use crate::ty::{
    self, Ty, TyCtxt, TypeFlags, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
};

/// Erases regions from `ty` to yield a new type.
fn erase_and_anonymize_regions_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    if let Some(erased_ty) = tcx.erase_and_anonymize_regions_ty_cache.get(&ty) {
        return erased_ty;
    }

    // N.B., use `super_fold_with` here. If we used `fold_with`, it
    // could invoke `erase_and_anonymize_regions_ty` recursively.
    let erased_ty = ty.super_fold_with(&mut RegionEraserAndAnonymizerVisitor { tcx });
    let old_ty = tcx.erase_and_anonymize_regions_ty_cache.insert(ty, erased_ty);
    // If two threads raced to erase the same type, they should agree.
    try { debug_assert_eq!(old_ty?, erased_ty) };
    erased_ty
}

impl<'tcx> TyCtxt<'tcx> {
    /// Returns an equivalent value with all free regions removed and
    /// bound regions anonymized. (note that bound regions are important
    /// for subtyping and generally type equality so *cannot* be removed)
    pub fn erase_and_anonymize_regions<T>(self, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        // If there's nothing to erase or anonymize, avoid performing the query at all
        if !value.has_type_flags(TypeFlags::HAS_BINDER_VARS | TypeFlags::HAS_FREE_REGIONS) {
            return value;
        }
        debug!("erase_and_anonymize_regions({:?})", value);
        let value1 = value.fold_with(&mut RegionEraserAndAnonymizerVisitor { tcx: self });
        debug!("erase_and_anonymize_regions = {:?}", value1);
        value1
    }
}

struct RegionEraserAndAnonymizerVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for RegionEraserAndAnonymizerVisitor<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !ty.has_type_flags(TypeFlags::HAS_BINDER_VARS | TypeFlags::HAS_FREE_REGIONS) {
            ty
        } else if ty.has_infer() {
            ty.super_fold_with(self)
        } else {
            erase_and_anonymize_regions_ty(self.tcx, ty)
        }
    }

    fn fold_binder<T>(&mut self, t: ty::Binder<'tcx, T>) -> ty::Binder<'tcx, T>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let u = self.tcx.anonymize_bound_vars(t);
        u.super_fold_with(self)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        // We must not erase bound regions. `for<'a> fn(&'a ())` and
        // `fn(&'free ())` are different types: they may implement different
        // traits and have a different `TypeId`.
        match r.kind() {
            ty::ReBound(..) => r,
            _ => self.tcx.lifetimes.re_erased,
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if ct.has_type_flags(TypeFlags::HAS_BINDER_VARS | TypeFlags::HAS_FREE_REGIONS) {
            ct.super_fold_with(self)
        } else {
            ct
        }
    }

    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        if p.has_type_flags(TypeFlags::HAS_BINDER_VARS | TypeFlags::HAS_FREE_REGIONS) {
            p.super_fold_with(self)
        } else {
            p
        }
    }

    fn fold_clauses(&mut self, c: ty::Clauses<'tcx>) -> ty::Clauses<'tcx> {
        if c.has_type_flags(TypeFlags::HAS_BINDER_VARS | TypeFlags::HAS_FREE_REGIONS) {
            c.super_fold_with(self)
        } else {
            c
        }
    }
}
