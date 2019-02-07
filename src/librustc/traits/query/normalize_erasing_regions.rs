//! Methods for normalizing when you don't care about regions (and
//! aren't doing type inference). If either of those things don't
//! apply to you, use `infcx.normalize(...)`.
//!
//! The methods in this file use a `TypeFolder` to recursively process
//! contents, invoking the underlying
//! `normalize_ty_after_erasing_regions` query for each type found
//! within. (This underlying query is what is cached.)

use crate::ty::{self, Ty, TyCtxt};
use crate::ty::fold::{TypeFoldable, TypeFolder};

impl<'cx, 'tcx> TyCtxt<'cx, 'tcx, 'tcx> {
    /// Erase the regions in `value` and then fully normalize all the
    /// types found within. The result will also have regions erased.
    ///
    /// This is appropriate to use only after type-check: it assumes
    /// that normalization will succeed, for example.
    pub fn normalize_erasing_regions<T>(self, param_env: ty::ParamEnv<'tcx>, value: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        debug!(
            "normalize_erasing_regions::<{}>(value={:?}, param_env={:?})",
            unsafe { ::std::intrinsics::type_name::<T>() },
            value,
            param_env,
        );

        // Erase first before we do the real query -- this keeps the
        // cache from being too polluted.
        let value = self.erase_regions(&value);
        if !value.has_projections() {
            value
        } else {
            let Ok(t) = value.fold_with(&mut NormalizeAfterErasingRegionsFolder {
                tcx: self,
                param_env: param_env,
            });
            t
        }
    }

    /// If you have a `Binder<T>`, you can do this to strip out the
    /// late-bound regions and then normalize the result, yielding up
    /// a `T` (with regions erased). This is appropriate when the
    /// binder is being instantiated at the call site.
    ///
    /// N.B., currently, higher-ranked type bounds inhibit
    /// normalization. Therefore, each time we erase them in
    /// codegen, we need to normalize the contents.
    pub fn normalize_erasing_late_bound_regions<T>(
        self,
        param_env: ty::ParamEnv<'tcx>,
        value: &ty::Binder<T>,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        assert!(!value.needs_subst());
        let value = self.erase_late_bound_regions(value);
        self.normalize_erasing_regions(param_env, value)
    }
}

struct NormalizeAfterErasingRegionsFolder<'cx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'cx, 'tcx> TypeFolder<'tcx, 'tcx> for NormalizeAfterErasingRegionsFolder<'cx, 'tcx> {
    type Error = !;

    fn tcx<'a>(&'a self) -> TyCtxt<'a, 'tcx, 'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Result<Ty<'tcx>, !> {
        Ok(self.tcx.normalize_ty_after_erasing_regions(self.param_env.and(ty)))
    }
}
