//! Various extensions traits for Chalk types.

use hir_def::{ItemContainerId, Lookup, TraitId};

use crate::{
    Interner, ProjectionTy, Substitution, TraitRef, Ty, db::HirDatabase, from_assoc_type_id,
    from_chalk_trait_id, generics::generics, to_chalk_trait_id,
};

pub(crate) trait ProjectionTyExt {
    fn trait_ref(&self, db: &dyn HirDatabase) -> TraitRef;
    fn trait_(&self, db: &dyn HirDatabase) -> TraitId;
    fn self_type_parameter(&self, db: &dyn HirDatabase) -> Ty;
}

impl ProjectionTyExt for ProjectionTy {
    fn trait_ref(&self, db: &dyn HirDatabase) -> TraitRef {
        // FIXME: something like `Split` trait from chalk-solve might be nice.
        let generics = generics(db, from_assoc_type_id(self.associated_ty_id).into());
        let parent_len = generics.parent_generics().map_or(0, |g| g.len_self());
        let substitution =
            Substitution::from_iter(Interner, self.substitution.iter(Interner).take(parent_len));
        TraitRef { trait_id: to_chalk_trait_id(self.trait_(db)), substitution }
    }

    fn trait_(&self, db: &dyn HirDatabase) -> TraitId {
        match from_assoc_type_id(self.associated_ty_id).lookup(db).container {
            ItemContainerId::TraitId(it) => it,
            _ => panic!("projection ty without parent trait"),
        }
    }

    fn self_type_parameter(&self, db: &dyn HirDatabase) -> Ty {
        self.trait_ref(db).self_type_parameter(Interner)
    }
}

pub(crate) trait TraitRefExt {
    fn hir_trait_id(&self) -> TraitId;
}

impl TraitRefExt for TraitRef {
    fn hir_trait_id(&self) -> TraitId {
        from_chalk_trait_id(self.trait_id)
    }
}
