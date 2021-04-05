//! Various extensions traits for Chalk types.

use hir_def::{AssocContainerId, Lookup, TraitId};

use crate::{
    db::HirDatabase, from_assoc_type_id, to_chalk_trait_id, Interner, ProjectionTy, TraitRef, Ty,
    TyKind,
};

pub trait TyExt {
    fn is_unit(&self) -> bool;
}

impl TyExt for Ty {
    fn is_unit(&self) -> bool {
        matches!(self.kind(&Interner), TyKind::Tuple(0, _))
    }
}

pub trait ProjectionTyExt {
    fn trait_ref(&self, db: &dyn HirDatabase) -> TraitRef;
    fn trait_(&self, db: &dyn HirDatabase) -> TraitId;
}

impl ProjectionTyExt for ProjectionTy {
    fn trait_ref(&self, db: &dyn HirDatabase) -> TraitRef {
        TraitRef {
            trait_id: to_chalk_trait_id(self.trait_(db)),
            substitution: self.substitution.clone(),
        }
    }

    fn trait_(&self, db: &dyn HirDatabase) -> TraitId {
        match from_assoc_type_id(self.associated_ty_id).lookup(db.upcast()).container {
            AssocContainerId::TraitId(it) => it,
            _ => panic!("projection ty without parent trait"),
        }
    }
}
