//! Various extensions traits for Chalk types.

use chalk_ir::{
    cast::Cast, FloatTy, IntTy, Mutability, Scalar, TyVariableKind, TypeOutlives, UintTy,
};
use hir_def::{
    builtin_type::{BuiltinFloat, BuiltinInt, BuiltinType, BuiltinUint},
    generics::TypeOrConstParamData,
    lang_item::LangItem,
    type_ref::Rawness,
    DefWithBodyId, FunctionId, GenericDefId, HasModule, ItemContainerId, Lookup, TraitId,
};

use crate::{
    db::HirDatabase, from_assoc_type_id, from_chalk_trait_id, from_foreign_def_id,
    from_placeholder_idx, generics::generics, to_chalk_trait_id, utils::ClosureSubst, AdtId,
    AliasEq, AliasTy, Binders, CallableDefId, CallableSig, Canonical, CanonicalVarKinds, ClosureId,
    DynTy, FnPointer, ImplTraitId, InEnvironment, Interner, Lifetime, ProjectionTy,
    QuantifiedWhereClause, Substitution, TraitRef, Ty, TyBuilder, TyKind, TypeFlags, WhereClause,
};

pub trait TyExt {
    fn is_unit(&self) -> bool;
    fn is_integral(&self) -> bool;
    fn is_scalar(&self) -> bool;
    fn is_floating_point(&self) -> bool;
    fn is_never(&self) -> bool;
    fn is_str(&self) -> bool;
    fn is_unknown(&self) -> bool;
    fn contains_unknown(&self) -> bool;
    fn is_ty_var(&self) -> bool;
    fn is_union(&self) -> bool;

    fn as_adt(&self) -> Option<(hir_def::AdtId, &Substitution)>;
    fn as_builtin(&self) -> Option<BuiltinType>;
    fn as_tuple(&self) -> Option<&Substitution>;
    fn as_closure(&self) -> Option<ClosureId>;
    fn as_fn_def(&self, db: &dyn HirDatabase) -> Option<FunctionId>;
    fn as_reference(&self) -> Option<(&Ty, Lifetime, Mutability)>;
    fn as_raw_ptr(&self) -> Option<(&Ty, Mutability)>;
    fn as_reference_or_ptr(&self) -> Option<(&Ty, Rawness, Mutability)>;
    fn as_generic_def(&self, db: &dyn HirDatabase) -> Option<GenericDefId>;

    fn callable_def(&self, db: &dyn HirDatabase) -> Option<CallableDefId>;
    fn callable_sig(&self, db: &dyn HirDatabase) -> Option<CallableSig>;

    fn strip_references(&self) -> &Ty;
    fn strip_reference(&self) -> &Ty;

    /// If this is a `dyn Trait`, returns that trait.
    fn dyn_trait(&self) -> Option<TraitId>;

    fn impl_trait_bounds(&self, db: &dyn HirDatabase) -> Option<Vec<QuantifiedWhereClause>>;
    fn associated_type_parent_trait(&self, db: &dyn HirDatabase) -> Option<TraitId>;
    fn is_copy(self, db: &dyn HirDatabase, owner: DefWithBodyId) -> bool;

    /// FIXME: Get rid of this, it's not a good abstraction
    fn equals_ctor(&self, other: &Ty) -> bool;
}

impl TyExt for Ty {
    fn is_unit(&self) -> bool {
        matches!(self.kind(Interner), TyKind::Tuple(0, _))
    }

    fn is_integral(&self) -> bool {
        matches!(
            self.kind(Interner),
            TyKind::Scalar(Scalar::Int(_) | Scalar::Uint(_))
                | TyKind::InferenceVar(_, TyVariableKind::Integer)
        )
    }

    fn is_scalar(&self) -> bool {
        matches!(self.kind(Interner), TyKind::Scalar(_))
    }

    fn is_floating_point(&self) -> bool {
        matches!(
            self.kind(Interner),
            TyKind::Scalar(Scalar::Float(_)) | TyKind::InferenceVar(_, TyVariableKind::Float)
        )
    }

    fn is_never(&self) -> bool {
        matches!(self.kind(Interner), TyKind::Never)
    }

    fn is_str(&self) -> bool {
        matches!(self.kind(Interner), TyKind::Str)
    }

    fn is_unknown(&self) -> bool {
        matches!(self.kind(Interner), TyKind::Error)
    }

    fn contains_unknown(&self) -> bool {
        self.data(Interner).flags.contains(TypeFlags::HAS_ERROR)
    }

    fn is_ty_var(&self) -> bool {
        matches!(self.kind(Interner), TyKind::InferenceVar(_, _))
    }

    fn is_union(&self) -> bool {
        matches!(self.adt_id(Interner), Some(AdtId(hir_def::AdtId::UnionId(_))))
    }

    fn as_adt(&self) -> Option<(hir_def::AdtId, &Substitution)> {
        match self.kind(Interner) {
            TyKind::Adt(AdtId(adt), parameters) => Some((*adt, parameters)),
            _ => None,
        }
    }

    fn as_builtin(&self) -> Option<BuiltinType> {
        match self.kind(Interner) {
            TyKind::Str => Some(BuiltinType::Str),
            TyKind::Scalar(Scalar::Bool) => Some(BuiltinType::Bool),
            TyKind::Scalar(Scalar::Char) => Some(BuiltinType::Char),
            TyKind::Scalar(Scalar::Float(fty)) => Some(BuiltinType::Float(match fty {
                FloatTy::F128 => BuiltinFloat::F128,
                FloatTy::F64 => BuiltinFloat::F64,
                FloatTy::F32 => BuiltinFloat::F32,
                FloatTy::F16 => BuiltinFloat::F16,
            })),
            TyKind::Scalar(Scalar::Int(ity)) => Some(BuiltinType::Int(match ity {
                IntTy::Isize => BuiltinInt::Isize,
                IntTy::I8 => BuiltinInt::I8,
                IntTy::I16 => BuiltinInt::I16,
                IntTy::I32 => BuiltinInt::I32,
                IntTy::I64 => BuiltinInt::I64,
                IntTy::I128 => BuiltinInt::I128,
            })),
            TyKind::Scalar(Scalar::Uint(ity)) => Some(BuiltinType::Uint(match ity {
                UintTy::Usize => BuiltinUint::Usize,
                UintTy::U8 => BuiltinUint::U8,
                UintTy::U16 => BuiltinUint::U16,
                UintTy::U32 => BuiltinUint::U32,
                UintTy::U64 => BuiltinUint::U64,
                UintTy::U128 => BuiltinUint::U128,
            })),
            _ => None,
        }
    }

    fn as_tuple(&self) -> Option<&Substitution> {
        match self.kind(Interner) {
            TyKind::Tuple(_, substs) => Some(substs),
            _ => None,
        }
    }

    fn as_closure(&self) -> Option<ClosureId> {
        match self.kind(Interner) {
            TyKind::Closure(id, _) => Some(*id),
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

    fn as_raw_ptr(&self) -> Option<(&Ty, Mutability)> {
        match self.kind(Interner) {
            TyKind::Raw(mutability, ty) => Some((ty, *mutability)),
            _ => None,
        }
    }

    fn as_reference_or_ptr(&self) -> Option<(&Ty, Rawness, Mutability)> {
        match self.kind(Interner) {
            TyKind::Ref(mutability, _, ty) => Some((ty, Rawness::Ref, *mutability)),
            TyKind::Raw(mutability, ty) => Some((ty, Rawness::RawPtr, *mutability)),
            _ => None,
        }
    }

    fn as_generic_def(&self, db: &dyn HirDatabase) -> Option<GenericDefId> {
        match *self.kind(Interner) {
            TyKind::Adt(AdtId(adt), ..) => Some(adt.into()),
            TyKind::FnDef(callable, ..) => Some(GenericDefId::from_callable(
                db.upcast(),
                db.lookup_intern_callable_def(callable.into()),
            )),
            TyKind::AssociatedType(type_alias, ..) => Some(from_assoc_type_id(type_alias).into()),
            TyKind::Foreign(type_alias, ..) => Some(from_foreign_def_id(type_alias).into()),
            _ => None,
        }
    }

    fn callable_def(&self, db: &dyn HirDatabase) -> Option<CallableDefId> {
        match self.kind(Interner) {
            &TyKind::FnDef(def, ..) => Some(db.lookup_intern_callable_def(def.into())),
            _ => None,
        }
    }

    fn callable_sig(&self, db: &dyn HirDatabase) -> Option<CallableSig> {
        match self.kind(Interner) {
            TyKind::Function(fn_ptr) => Some(CallableSig::from_fn_ptr(fn_ptr)),
            TyKind::FnDef(def, parameters) => Some(CallableSig::from_def(db, *def, parameters)),
            TyKind::Closure(.., substs) => ClosureSubst(substs).sig_ty().callable_sig(db),
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

    fn strip_reference(&self) -> &Ty {
        self.as_reference().map_or(self, |(ty, _, _)| ty)
    }

    fn impl_trait_bounds(&self, db: &dyn HirDatabase) -> Option<Vec<QuantifiedWhereClause>> {
        match self.kind(Interner) {
            TyKind::OpaqueType(opaque_ty_id, subst) => {
                match db.lookup_intern_impl_trait_id((*opaque_ty_id).into()) {
                    ImplTraitId::AsyncBlockTypeImplTrait(def, _expr) => {
                        let krate = def.module(db.upcast()).krate();
                        if let Some(future_trait) =
                            db.lang_item(krate, LangItem::Future).and_then(|item| item.as_trait())
                        {
                            // This is only used by type walking.
                            // Parameters will be walked outside, and projection predicate is not used.
                            // So just provide the Future trait.
                            let impl_bound = Binders::empty(
                                Interner,
                                WhereClause::Implemented(TraitRef {
                                    trait_id: to_chalk_trait_id(future_trait),
                                    substitution: Substitution::empty(Interner),
                                }),
                            );
                            Some(vec![impl_bound])
                        } else {
                            None
                        }
                    }
                    ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                        db.return_type_impl_traits(func).map(|it| {
                            let data =
                                (*it).as_ref().map(|rpit| rpit.impl_traits[idx].bounds.clone());
                            data.substitute(Interner, &subst).into_value_and_skipped_binders().0
                        })
                    }
                    ImplTraitId::TypeAliasImplTrait(alias, idx) => {
                        db.type_alias_impl_traits(alias).map(|it| {
                            let data =
                                (*it).as_ref().map(|rpit| rpit.impl_traits[idx].bounds.clone());
                            data.substitute(Interner, &subst).into_value_and_skipped_binders().0
                        })
                    }
                }
            }
            TyKind::Alias(AliasTy::Opaque(opaque_ty)) => {
                let predicates = match db.lookup_intern_impl_trait_id(opaque_ty.opaque_ty_id.into())
                {
                    ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                        db.return_type_impl_traits(func).map(|it| {
                            let data =
                                (*it).as_ref().map(|rpit| rpit.impl_traits[idx].bounds.clone());
                            data.substitute(Interner, &opaque_ty.substitution)
                        })
                    }
                    ImplTraitId::TypeAliasImplTrait(alias, idx) => {
                        db.type_alias_impl_traits(alias).map(|it| {
                            let data =
                                (*it).as_ref().map(|rpit| rpit.impl_traits[idx].bounds.clone());
                            data.substitute(Interner, &opaque_ty.substitution)
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
                let param_data = &generic_params[id.local_id];
                match param_data {
                    TypeOrConstParamData::TypeParamData(p) => match p.provenance {
                        hir_def::generics::TypeParamProvenance::ArgumentImplTrait => {
                            let substs = TyBuilder::placeholder_subst(db, id.parent);
                            let predicates = db
                                .generic_predicates(id.parent)
                                .iter()
                                .map(|pred| pred.clone().substitute(Interner, &substs))
                                .filter(|wc| match wc.skip_binders() {
                                    WhereClause::Implemented(tr) => {
                                        &tr.self_type_parameter(Interner) == self
                                    }
                                    WhereClause::AliasEq(AliasEq {
                                        alias: AliasTy::Projection(proj),
                                        ty: _,
                                    }) => &proj.self_type_parameter(db) == self,
                                    WhereClause::TypeOutlives(TypeOutlives { ty, lifetime: _ }) => {
                                        ty == self
                                    }
                                    _ => false,
                                })
                                .collect::<Vec<_>>();

                            Some(predicates)
                        }
                        _ => None,
                    },
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn associated_type_parent_trait(&self, db: &dyn HirDatabase) -> Option<TraitId> {
        match self.kind(Interner) {
            TyKind::AssociatedType(id, ..) => {
                match from_assoc_type_id(*id).lookup(db.upcast()).container {
                    ItemContainerId::TraitId(trait_id) => Some(trait_id),
                    _ => None,
                }
            }
            TyKind::Alias(AliasTy::Projection(projection_ty)) => {
                match from_assoc_type_id(projection_ty.associated_ty_id)
                    .lookup(db.upcast())
                    .container
                {
                    ItemContainerId::TraitId(trait_id) => Some(trait_id),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn is_copy(self, db: &dyn HirDatabase, owner: DefWithBodyId) -> bool {
        let crate_id = owner.module(db.upcast()).krate();
        let Some(copy_trait) = db.lang_item(crate_id, LangItem::Copy).and_then(|it| it.as_trait())
        else {
            return false;
        };
        let trait_ref = TyBuilder::trait_ref(db, copy_trait).push(self).build();
        let env = db.trait_environment_for_body(owner);
        let goal = Canonical {
            value: InEnvironment::new(&env.env, trait_ref.cast(Interner)),
            binders: CanonicalVarKinds::empty(Interner),
        };
        db.trait_solve(crate_id, None, goal).is_some()
    }

    fn equals_ctor(&self, other: &Ty) -> bool {
        match (self.kind(Interner), other.kind(Interner)) {
            (TyKind::Adt(adt, ..), TyKind::Adt(adt2, ..)) => adt == adt2,
            (TyKind::Slice(_), TyKind::Slice(_)) | (TyKind::Array(_, _), TyKind::Array(_, _)) => {
                true
            }
            (TyKind::FnDef(def_id, ..), TyKind::FnDef(def_id2, ..)) => def_id == def_id2,
            (TyKind::OpaqueType(ty_id, ..), TyKind::OpaqueType(ty_id2, ..)) => ty_id == ty_id2,
            (TyKind::AssociatedType(ty_id, ..), TyKind::AssociatedType(ty_id2, ..)) => {
                ty_id == ty_id2
            }
            (TyKind::Foreign(ty_id, ..), TyKind::Foreign(ty_id2, ..)) => ty_id == ty_id2,
            (TyKind::Closure(id1, _), TyKind::Closure(id2, _)) => id1 == id2,
            (TyKind::Ref(mutability, ..), TyKind::Ref(mutability2, ..))
            | (TyKind::Raw(mutability, ..), TyKind::Raw(mutability2, ..)) => {
                mutability == mutability2
            }
            (
                TyKind::Function(FnPointer { num_binders, sig, .. }),
                TyKind::Function(FnPointer { num_binders: num_binders2, sig: sig2, .. }),
            ) => num_binders == num_binders2 && sig == sig2,
            (TyKind::Tuple(cardinality, _), TyKind::Tuple(cardinality2, _)) => {
                cardinality == cardinality2
            }
            (TyKind::Str, TyKind::Str) | (TyKind::Never, TyKind::Never) => true,
            (TyKind::Scalar(scalar), TyKind::Scalar(scalar2)) => scalar == scalar2,
            _ => false,
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
        let generics = generics(db.upcast(), from_assoc_type_id(self.associated_ty_id).into());
        let substitution = Substitution::from_iter(
            Interner,
            self.substitution.iter(Interner).skip(generics.len_self()),
        );
        TraitRef { trait_id: to_chalk_trait_id(self.trait_(db)), substitution }
    }

    fn trait_(&self, db: &dyn HirDatabase) -> TraitId {
        match from_assoc_type_id(self.associated_ty_id).lookup(db.upcast()).container {
            ItemContainerId::TraitId(it) => it,
            _ => panic!("projection ty without parent trait"),
        }
    }

    fn self_type_parameter(&self, db: &dyn HirDatabase) -> Ty {
        self.trait_ref(db).self_type_parameter(Interner)
    }
}

pub trait DynTyExt {
    fn principal(&self) -> Option<&TraitRef>;
}

impl DynTyExt for DynTy {
    fn principal(&self) -> Option<&TraitRef> {
        self.bounds.skip_binders().interned().first().and_then(|b| match b.skip_binders() {
            crate::WhereClause::Implemented(trait_ref) => Some(trait_ref),
            _ => None,
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
