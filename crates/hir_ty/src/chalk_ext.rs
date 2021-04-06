//! Various extensions traits for Chalk types.

use chalk_ir::Mutability;
use hir_def::{
    type_ref::Rawness, AssocContainerId, FunctionId, GenericDefId, HasModule, Lookup, TraitId,
};

use crate::{
    db::HirDatabase, from_assoc_type_id, from_chalk_trait_id, from_foreign_def_id,
    from_placeholder_idx, to_chalk_trait_id, AdtId, AliasEq, AliasTy, Binders, CallableDefId,
    CallableSig, ImplTraitId, Interner, Lifetime, ProjectionTy, QuantifiedWhereClause,
    Substitution, TraitRef, Ty, TyBuilder, TyKind, WhereClause,
};

pub trait TyExt {
    fn is_unit(&self) -> bool;
    fn is_never(&self) -> bool;
    fn is_unknown(&self) -> bool;

    fn as_adt(&self) -> Option<(hir_def::AdtId, &Substitution)>;
    fn as_tuple(&self) -> Option<&Substitution>;
    fn as_fn_def(&self, db: &dyn HirDatabase) -> Option<FunctionId>;
    fn as_reference(&self) -> Option<(&Ty, Lifetime, Mutability)>;
    fn as_reference_or_ptr(&self) -> Option<(&Ty, Rawness, Mutability)>;
    fn as_generic_def(&self, db: &dyn HirDatabase) -> Option<GenericDefId>;

    fn callable_def(&self, db: &dyn HirDatabase) -> Option<CallableDefId>;
    fn callable_sig(&self, db: &dyn HirDatabase) -> Option<CallableSig>;

    fn strip_references(&self) -> &Ty;

    /// If this is a `dyn Trait`, returns that trait.
    fn dyn_trait(&self) -> Option<TraitId>;

    fn impl_trait_bounds(&self, db: &dyn HirDatabase) -> Option<Vec<QuantifiedWhereClause>>;
    fn associated_type_parent_trait(&self, db: &dyn HirDatabase) -> Option<TraitId>;
}

impl TyExt for Ty {
    fn is_unit(&self) -> bool {
        matches!(self.kind(&Interner), TyKind::Tuple(0, _))
    }

    fn is_never(&self) -> bool {
        matches!(self.kind(&Interner), TyKind::Never)
    }

    fn is_unknown(&self) -> bool {
        matches!(self.kind(&Interner), TyKind::Error)
    }

    fn as_adt(&self) -> Option<(hir_def::AdtId, &Substitution)> {
        match self.kind(&Interner) {
            TyKind::Adt(AdtId(adt), parameters) => Some((*adt, parameters)),
            _ => None,
        }
    }

    fn as_tuple(&self) -> Option<&Substitution> {
        match self.kind(&Interner) {
            TyKind::Tuple(_, substs) => Some(substs),
            _ => None,
        }
    }

    fn as_fn_def(&self, db: &dyn HirDatabase) -> Option<FunctionId> {
        if let Some(CallableDefId::FunctionId(func)) = self.callable_def(db) {
            Some(func)
        } else {
            None
        }
    }
    fn as_reference(&self) -> Option<(&Ty, Lifetime, Mutability)> {
        match self.kind(&Interner) {
            TyKind::Ref(mutability, lifetime, ty) => Some((ty, *lifetime, *mutability)),
            _ => None,
        }
    }

    fn as_reference_or_ptr(&self) -> Option<(&Ty, Rawness, Mutability)> {
        match self.kind(&Interner) {
            TyKind::Ref(mutability, _, ty) => Some((ty, Rawness::Ref, *mutability)),
            TyKind::Raw(mutability, ty) => Some((ty, Rawness::RawPtr, *mutability)),
            _ => None,
        }
    }

    fn as_generic_def(&self, db: &dyn HirDatabase) -> Option<GenericDefId> {
        match *self.kind(&Interner) {
            TyKind::Adt(AdtId(adt), ..) => Some(adt.into()),
            TyKind::FnDef(callable, ..) => {
                Some(db.lookup_intern_callable_def(callable.into()).into())
            }
            TyKind::AssociatedType(type_alias, ..) => Some(from_assoc_type_id(type_alias).into()),
            TyKind::Foreign(type_alias, ..) => Some(from_foreign_def_id(type_alias).into()),
            _ => None,
        }
    }

    fn callable_def(&self, db: &dyn HirDatabase) -> Option<CallableDefId> {
        match self.kind(&Interner) {
            &TyKind::FnDef(def, ..) => Some(db.lookup_intern_callable_def(def.into())),
            _ => None,
        }
    }

    fn callable_sig(&self, db: &dyn HirDatabase) -> Option<CallableSig> {
        match self.kind(&Interner) {
            TyKind::Function(fn_ptr) => Some(CallableSig::from_fn_ptr(fn_ptr)),
            TyKind::FnDef(def, parameters) => {
                let callable_def = db.lookup_intern_callable_def((*def).into());
                let sig = db.callable_item_signature(callable_def);
                Some(sig.substitute(&Interner, &parameters))
            }
            TyKind::Closure(.., substs) => {
                let sig_param = substs.at(&Interner, 0).assert_ty_ref(&Interner);
                sig_param.callable_sig(db)
            }
            _ => None,
        }
    }

    fn dyn_trait(&self) -> Option<TraitId> {
        let trait_ref = match self.kind(&Interner) {
            TyKind::Dyn(dyn_ty) => dyn_ty.bounds.skip_binders().interned().get(0).and_then(|b| {
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
        while let TyKind::Ref(_mutability, _lifetime, ty) = t.kind(&Interner) {
            t = ty;
        }
        t
    }

    fn impl_trait_bounds(&self, db: &dyn HirDatabase) -> Option<Vec<QuantifiedWhereClause>> {
        match self.kind(&Interner) {
            TyKind::OpaqueType(opaque_ty_id, ..) => {
                match db.lookup_intern_impl_trait_id((*opaque_ty_id).into()) {
                    ImplTraitId::AsyncBlockTypeImplTrait(def, _expr) => {
                        let krate = def.module(db.upcast()).krate();
                        if let Some(future_trait) = db
                            .lang_item(krate, "future_trait".into())
                            .and_then(|item| item.as_trait())
                        {
                            // This is only used by type walking.
                            // Parameters will be walked outside, and projection predicate is not used.
                            // So just provide the Future trait.
                            let impl_bound = Binders::empty(
                                &Interner,
                                WhereClause::Implemented(TraitRef {
                                    trait_id: to_chalk_trait_id(future_trait),
                                    substitution: Substitution::empty(&Interner),
                                }),
                            );
                            Some(vec![impl_bound])
                        } else {
                            None
                        }
                    }
                    ImplTraitId::ReturnTypeImplTrait(..) => None,
                }
            }
            TyKind::Alias(AliasTy::Opaque(opaque_ty)) => {
                let predicates = match db.lookup_intern_impl_trait_id(opaque_ty.opaque_ty_id.into())
                {
                    ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                        db.return_type_impl_traits(func).map(|it| {
                            let data = (*it)
                                .as_ref()
                                .map(|rpit| rpit.impl_traits[idx as usize].bounds.clone());
                            data.substitute(&Interner, &opaque_ty.substitution)
                        })
                    }
                    // It always has an parameter for Future::Output type.
                    ImplTraitId::AsyncBlockTypeImplTrait(..) => unreachable!(),
                };

                predicates.map(|it| it.into_value_and_skipped_binders().0)
            }
            TyKind::Placeholder(idx) => {
                let id = from_placeholder_idx(db, *idx);
                let generic_params = db.generic_params(id.parent);
                let param_data = &generic_params.types[id.local_id];
                match param_data.provenance {
                    hir_def::generics::TypeParamProvenance::ArgumentImplTrait => {
                        let substs = TyBuilder::type_params_subst(db, id.parent);
                        let predicates = db
                            .generic_predicates(id.parent)
                            .into_iter()
                            .map(|pred| pred.clone().substitute(&Interner, &substs))
                            .filter(|wc| match &wc.skip_binders() {
                                WhereClause::Implemented(tr) => {
                                    tr.self_type_parameter(&Interner) == self
                                }
                                WhereClause::AliasEq(AliasEq {
                                    alias: AliasTy::Projection(proj),
                                    ty: _,
                                }) => proj.self_type_parameter(&Interner) == self,
                                _ => false,
                            })
                            .collect::<Vec<_>>();

                        Some(predicates)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn associated_type_parent_trait(&self, db: &dyn HirDatabase) -> Option<TraitId> {
        match self.kind(&Interner) {
            TyKind::AssociatedType(id, ..) => {
                match from_assoc_type_id(*id).lookup(db.upcast()).container {
                    AssocContainerId::TraitId(trait_id) => Some(trait_id),
                    _ => None,
                }
            }
            TyKind::Alias(AliasTy::Projection(projection_ty)) => {
                match from_assoc_type_id(projection_ty.associated_ty_id)
                    .lookup(db.upcast())
                    .container
                {
                    AssocContainerId::TraitId(trait_id) => Some(trait_id),
                    _ => None,
                }
            }
            _ => None,
        }
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
