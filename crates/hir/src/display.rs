//! HirDisplay implementations for various hir types.
use hir_ty::display::{
    write_bounds_like_dyn_trait_with_prefix, HirDisplay, HirDisplayError, HirFormatter,
};

use crate::{Substs, Type, TypeParam};

impl HirDisplay for Type {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        self.ty.value.hir_fmt(f)
    }
}

impl HirDisplay for TypeParam {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        write!(f, "{}", self.name(f.db))?;
        let bounds = f.db.generic_predicates_for_param(self.id);
        let substs = Substs::type_params(f.db, self.id.parent);
        let predicates = bounds.iter().cloned().map(|b| b.subst(&substs)).collect::<Vec<_>>();
        if !(predicates.is_empty() || f.omit_verbose_types()) {
            write_bounds_like_dyn_trait_with_prefix(":", &predicates, f)?;
        }
        Ok(())
    }
}
