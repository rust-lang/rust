//! Methods for normalizing when you don't care about regions (and
//! aren't doing type inference). If either of those things don't
//! apply to you, use `infcx.normalize(...)`.
//!
//! The methods in this file use a `TypeFolder` to recursively process
//! contents, invoking the underlying
//! `normalize_generic_arg_after_erasing_regions` query for each type
//! or constant found within. (This underlying query is what is cached.)

use crate::ty::fold::{TypeFoldable, TypeFolder};
use crate::ty::subst::{Subst, SubstsRef};
use crate::ty::{self, Ty, TyCtxt};

impl<'tcx> TyCtxt<'tcx> {
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
            std::any::type_name::<T>(),
            value,
            param_env,
        );

        // Erase first before we do the real query -- this keeps the
        // cache from being too polluted.
        let value = self.erase_regions(value);
        if !value.has_projections() {
            value
        } else {
            value.fold_with(&mut NormalizeAfterErasingRegionsFolder { tcx: self, param_env })
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
        value: ty::Binder<T>,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        let value = self.erase_late_bound_regions(value);
        self.normalize_erasing_regions(param_env, value)
    }

    /// Monomorphizes a type from the AST by first applying the
    /// in-scope substitutions and then normalizing any associated
    /// types.
    pub fn subst_and_normalize_erasing_regions<T>(
        self,
        param_substs: SubstsRef<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: T,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        debug!(
            "subst_and_normalize_erasing_regions(\
             param_substs={:?}, \
             value={:?}, \
             param_env={:?})",
            param_substs, value, param_env,
        );
        let substituted = value.subst(self, param_substs);
        self.normalize_erasing_regions(param_env, substituted)
    }
}

struct NormalizeAfterErasingRegionsFolder<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

impl TypeFolder<'tcx> for NormalizeAfterErasingRegionsFolder<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let arg = self.param_env.and(ty.into());
        self.tcx.normalize_generic_arg_after_erasing_regions(arg).expect_ty()
    }

    fn fold_const(&mut self, c: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        let arg = self.param_env.and(c.into());
        self.tcx.normalize_generic_arg_after_erasing_regions(arg).expect_const()
    }
}
