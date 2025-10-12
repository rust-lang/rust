//! Various extensions traits for Chalk types.

use hir_def::{ItemContainerId, Lookup, TraitId};

use crate::{
    Binders, CallableSig, DynTy, Interner, ProjectionTy, Substitution, TraitRef, Ty, TyKind,
    db::HirDatabase, from_assoc_type_id, from_chalk_trait_id, generics::generics,
    to_chalk_trait_id, utils::ClosureSubst,
};

pub(crate) trait TyExt {
    fn is_unit(&self) -> bool;
    fn is_unknown(&self) -> bool;

    fn as_tuple(&self) -> Option<&Substitution>;

    fn callable_sig(&self, db: &dyn HirDatabase) -> Option<CallableSig>;
}

impl TyExt for Ty {
    fn is_unit(&self) -> bool {
        matches!(self.kind(Interner), TyKind::Tuple(0, _))
    }

    fn is_unknown(&self) -> bool {
        matches!(self.kind(Interner), TyKind::Error)
    }

    fn as_tuple(&self) -> Option<&Substitution> {
        match self.kind(Interner) {
            TyKind::Tuple(_, substs) => Some(substs),
            _ => None,
        }
    }

    fn callable_sig(&self, db: &dyn HirDatabase) -> Option<CallableSig> {
        match self.kind(Interner) {
            TyKind::Function(fn_ptr) => Some(CallableSig::from_fn_ptr(fn_ptr)),
            TyKind::FnDef(def, parameters) => Some(CallableSig::from_def(db, *def, parameters)),
            TyKind::Closure(.., substs) => ClosureSubst(substs).sig_ty(db).callable_sig(db),
            _ => None,
        }
    }
}

pub trait ProjectionTyExt {
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

pub(crate) trait DynTyExt {
    fn principal(&self) -> Option<Binders<Binders<&TraitRef>>>;
}

impl DynTyExt for DynTy {
    fn principal(&self) -> Option<Binders<Binders<&TraitRef>>> {
        self.bounds.as_ref().filter_map(|bounds| {
            bounds.interned().first().and_then(|b| {
                b.as_ref().filter_map(|b| match b {
                    crate::WhereClause::Implemented(trait_ref) => Some(trait_ref),
                    _ => None,
                })
            })
        })
    }
}

pub trait TraitRefExt {
    fn hir_trait_id(&self) -> TraitId;
}

impl TraitRefExt for TraitRef {
    fn hir_trait_id(&self) -> TraitId {
        from_chalk_trait_id(self.trait_id)
    }
}
