// This file contains various trait resolution methods used by codegen.
// They all assume regions can be erased and monomorphic types.  It
// seems likely that they should eventually be merged into more
// general routines.

use crate::ty::fold::TypeFoldable;
use crate::ty::subst::{Subst, SubstsRef};
use crate::ty::{self, TyCtxt};

impl<'tcx> TyCtxt<'tcx> {
    /// Monomorphizes a type from the AST by first applying the
    /// in-scope substitutions and then normalizing any associated
    /// types.
    pub fn subst_and_normalize_erasing_regions<T>(
        self,
        param_substs: SubstsRef<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: &T,
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
