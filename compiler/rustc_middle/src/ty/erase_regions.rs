use crate::ty::fold::{TypeFoldable, TypeFolder, TypeSuperFoldable};
use crate::ty::{self, Ty, TyCtxt, TypeFlags, TypeVisitableExt};

pub(super) fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers { erase_regions_ty, ..*providers };
}

fn erase_regions_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    // N.B., use `super_fold_with` here. If we used `fold_with`, it
    // could invoke the `erase_regions_ty` query recursively.
    ty.super_fold_with(&mut RegionEraserVisitor { tcx })
}

impl<'tcx> TyCtxt<'tcx> {
    /// Returns an equivalent value with all free regions removed (note
    /// that late-bound regions remain, because they are important for
    /// subtyping, but they are anonymized and normalized as well)..
    pub fn erase_regions<T>(self, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        // If there's nothing to erase avoid performing the query at all
        if !value.has_type_flags(TypeFlags::HAS_LATE_BOUND | TypeFlags::HAS_FREE_REGIONS) {
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
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if ty.needs_infer() { ty.super_fold_with(self) } else { self.tcx.erase_regions_ty(ty) }
    }

    fn fold_binder<T>(&mut self, t: ty::Binder<'tcx, T>) -> ty::Binder<'tcx, T>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let u = self.tcx.anonymize_bound_vars(t);
        u.super_fold_with(self)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        // because late-bound regions affect subtyping, we can't
        // erase the bound/free distinction, but we can replace
        // all free regions with 'erased.
        //
        // Note that we *CAN* replace early-bound regions -- the
        // type system never "sees" those, they get substituted
        // away. In codegen, they will always be erased to 'erased
        // whenever a substitution occurs.
        match *r {
            ty::ReLateBound(..) => r,
            _ => self.tcx.lifetimes.re_erased,
        }
    }
}
