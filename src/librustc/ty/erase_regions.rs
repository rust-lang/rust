use crate::ty::{self, Ty, TyCtxt, TypeFlags};
use crate::ty::fold::{TypeFolder, TypeFoldable};

pub(super) fn provide(providers: &mut ty::query::Providers<'_>) {
    *providers = ty::query::Providers {
        erase_regions_ty,
        ..*providers
    };
}

fn erase_regions_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    // N.B., use `super_fold_with` here. If we used `fold_with`, it
    // could invoke the `erase_regions_ty` query recursively.
    ty.super_fold_with(&mut RegionEraserVisitor { tcx }).unwrap_or_else(|e: !| e)
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    /// Returns an equivalent value with all free regions removed (note
    /// that late-bound regions remain, because they are important for
    /// subtyping, but they are anonymized and normalized as well)..
    pub fn erase_regions<T>(self, value: &T) -> T
        where T : TypeFoldable<'tcx>
    {
        // If there's nothing to erase avoid performing the query at all
        if !value.has_type_flags(TypeFlags::HAS_RE_LATE_BOUND | TypeFlags::HAS_FREE_REGIONS) {
            return value.clone();
        }

        let Ok(value1) = value.fold_with(&mut RegionEraserVisitor { tcx: self });
        debug!("erase_regions({:?}) = {:?}", value, value1);
        value1
    }
}

struct RegionEraserVisitor<'a, 'gcx: 'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for RegionEraserVisitor<'a, 'gcx, 'tcx> {
    type Error = !;

    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Result<Ty<'tcx>, !> {
        if let Some(ty_lifted) = self.tcx.lift_to_global(&ty) {
            Ok(self.tcx.erase_regions_ty(ty_lifted))
        } else {
            ty.super_fold_with(self)
        }
    }

    fn fold_binder<T>(&mut self, t: &ty::Binder<T>) -> Result<ty::Binder<T>, !>
        where T : TypeFoldable<'tcx>
    {
        let u = self.tcx.anonymize_late_bound_regions(t);
        u.super_fold_with(self)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> Result<ty::Region<'tcx>, !> {
        // because late-bound regions affect subtyping, we can't
        // erase the bound/free distinction, but we can replace
        // all free regions with 'erased.
        //
        // Note that we *CAN* replace early-bound regions -- the
        // type system never "sees" those, they get substituted
        // away. In codegen, they will always be erased to 'erased
        // whenever a substitution occurs.
        Ok(match *r {
            ty::ReLateBound(..) => r,
            _ => self.tcx.types.re_erased
        })
    }
}
