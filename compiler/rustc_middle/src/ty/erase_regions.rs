use tracing::debug;

use crate::ty::{
    self, Ty, TyCtxt, TypeFlags, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
};

fn erase_regions_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    if let Some(erased) = tcx.erased_ty_cache.lock().get(&ty) {
        *erased
    } else {
        let erased = ty.super_fold_with(&mut RegionEraserVisitor { tcx });
        tcx.erased_ty_cache.lock().insert(ty, erased);
        erased
    }
}

impl<'tcx> TyCtxt<'tcx> {
    /// Returns an equivalent value with all free regions removed (note
    /// that late-bound regions remain, because they are important for
    /// subtyping, but they are anonymized and normalized as well)..
    pub fn erase_regions<T>(self, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        // If there's nothing to erase or anonymize, avoid performing the query at all
        if !value.has_type_flags(TypeFlags::HAS_BINDER_VARS | TypeFlags::HAS_FREE_REGIONS) {
            return value;
        }
        debug!("erase_regions({:?})", value);
        let value1 = value.fold_with(&mut RegionEraserVisitor { tcx: self });
        debug!("erase_regions = {:?}", value1);
        value1
    }
}

struct RegionEraserVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for RegionEraserVisitor<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !ty.has_type_flags(TypeFlags::HAS_BINDER_VARS | TypeFlags::HAS_FREE_REGIONS) {
            ty
        } else {
            erase_regions_ty(self.tcx, ty)
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
}
