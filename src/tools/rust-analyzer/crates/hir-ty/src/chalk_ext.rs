//! Various extensions traits for Chalk types.

use chalk_ir::Mutability;
use hir_def::{FunctionId, ItemContainerId, Lookup, TraitId};

use crate::{
    AdtId, Binders, CallableDefId, CallableSig, DynTy, Interner, Lifetime, ProjectionTy,
    Substitution, ToChalk, TraitRef, Ty, TyKind, TypeFlags, WhereClause, db::HirDatabase,
    from_assoc_type_id, from_chalk_trait_id, generics::generics, to_chalk_trait_id,
    utils::ClosureSubst,
};

pub(crate) trait TyExt {
    fn is_unit(&self) -> bool;
    fn is_unknown(&self) -> bool;
    fn contains_unknown(&self) -> bool;

    fn as_adt(&self) -> Option<(hir_def::AdtId, &Substitution)>;
    fn as_tuple(&self) -> Option<&Substitution>;
    fn as_fn_def(&self, db: &dyn HirDatabase) -> Option<FunctionId>;
    fn as_reference(&self) -> Option<(&Ty, Lifetime, Mutability)>;

    fn callable_def(&self, db: &dyn HirDatabase) -> Option<CallableDefId>;
    fn callable_sig(&self, db: &dyn HirDatabase) -> Option<CallableSig>;

    fn strip_references(&self) -> &Ty;

    /// If this is a `dyn Trait`, returns that trait.
    fn dyn_trait(&self) -> Option<TraitId>;
}

impl TyExt for Ty {
    fn is_unit(&self) -> bool {
        matches!(self.kind(Interner), TyKind::Tuple(0, _))
    }

    fn is_unknown(&self) -> bool {
        matches!(self.kind(Interner), TyKind::Error)
    }

    fn contains_unknown(&self) -> bool {
        self.data(Interner).flags.contains(TypeFlags::HAS_ERROR)
    }

    fn as_adt(&self) -> Option<(hir_def::AdtId, &Substitution)> {
        match self.kind(Interner) {
            TyKind::Adt(AdtId(adt), parameters) => Some((*adt, parameters)),
            _ => None,
        }
    }

    fn as_tuple(&self) -> Option<&Substitution> {
        match self.kind(Interner) {
            TyKind::Tuple(_, substs) => Some(substs),
            _ => None,
        }
    }

    fn as_fn_def(&self, db: &dyn HirDatabase) -> Option<FunctionId> {
        match self.callable_def(db) {
            Some(CallableDefId::FunctionId(func)) => Some(func),
            Some(CallableDefId::StructId(_) | CallableDefId::EnumVariantId(_)) | None => None,
        }
    }

    fn as_reference(&self) -> Option<(&Ty, Lifetime, Mutability)> {
        match self.kind(Interner) {
            TyKind::Ref(mutability, lifetime, ty) => Some((ty, lifetime.clone(), *mutability)),
            _ => None,
        }
    }

    fn callable_def(&self, db: &dyn HirDatabase) -> Option<CallableDefId> {
        match self.kind(Interner) {
            &TyKind::FnDef(def, ..) => Some(ToChalk::from_chalk(db, def)),
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

    fn dyn_trait(&self) -> Option<TraitId> {
        let trait_ref = match self.kind(Interner) {
            // The principal trait bound should be the first element of the bounds. This is an
            // invariant ensured by `TyLoweringContext::lower_dyn_trait()`.
            // FIXME: dyn types may not have principal trait and we don't want to return auto trait
            // here.
            TyKind::Dyn(dyn_ty) => dyn_ty.bounds.skip_binders().interned().first().and_then(|b| {
                match b.skip_binders() {
                    WhereClause::Implemented(trait_ref) => Some(trait_ref),
                    _ => None,
                }
            }),
            _ => None,
        }?;
        Some(from_chalk_trait_id(trait_ref.trait_id))
    }

    fn strip_references(&self) -> &Ty {
        let mut t: &Ty = self;
        while let TyKind::Ref(_mutability, _lifetime, ty) = t.kind(Interner) {
            t = ty;
        }
        t
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
