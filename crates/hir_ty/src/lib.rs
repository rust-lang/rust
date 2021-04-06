//! The type system. We currently use this to infer types for completion, hover
//! information and various assists.
#[allow(unused)]
macro_rules! eprintln {
    ($($tt:tt)*) => { stdx::eprintln!($($tt)*) };
}

mod autoderef;
pub mod primitive;
pub mod traits;
pub mod method_resolution;
mod op;
mod lower;
pub(crate) mod infer;
pub(crate) mod utils;
mod chalk_cast;
mod chalk_ext;
mod builder;
mod walk;
mod types;

pub mod display;
pub mod db;
pub mod diagnostics;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod test_db;

use std::sync::Arc;

use itertools::Itertools;

use base_db::salsa;
use hir_def::{
    expr::ExprId, type_ref::Rawness, AssocContainerId, FunctionId, GenericDefId, HasModule,
    LifetimeParamId, Lookup, TraitId, TypeAliasId, TypeParamId,
};

use crate::{db::HirDatabase, display::HirDisplay, utils::generics};

pub use autoderef::autoderef;
pub use builder::TyBuilder;
pub use chalk_ext::{ProjectionTyExt, TyExt};
pub use infer::{could_unify, InferenceResult, InferenceVar};
pub use lower::{
    associated_type_shorthand_candidates, callable_item_sig, CallableDefId, ImplTraitLoweringMode,
    TyDefId, TyLoweringContext, ValueTyDefId,
};
pub use traits::{chalk::Interner, TraitEnvironment};
pub use types::*;
pub use walk::TypeWalk;

pub use chalk_ir::{
    cast::Cast, AdtId, BoundVar, DebruijnIndex, Mutability, Safety, Scalar, TyVariableKind,
};

pub type ForeignDefId = chalk_ir::ForeignDefId<Interner>;
pub type AssocTypeId = chalk_ir::AssocTypeId<Interner>;
pub type FnDefId = chalk_ir::FnDefId<Interner>;
pub type ClosureId = chalk_ir::ClosureId<Interner>;
pub type OpaqueTyId = chalk_ir::OpaqueTyId<Interner>;
pub type PlaceholderIndex = chalk_ir::PlaceholderIndex;

pub type VariableKind = chalk_ir::VariableKind<Interner>;
pub type VariableKinds = chalk_ir::VariableKinds<Interner>;
pub type CanonicalVarKinds = chalk_ir::CanonicalVarKinds<Interner>;

pub type Lifetime = chalk_ir::Lifetime<Interner>;
pub type LifetimeData = chalk_ir::LifetimeData<Interner>;
pub type LifetimeOutlives = chalk_ir::LifetimeOutlives<Interner>;

pub type ChalkTraitId = chalk_ir::TraitId<Interner>;

pub type FnSig = chalk_ir::FnSig<Interner>;

// FIXME: get rid of this
pub fn subst_prefix(s: &Substitution, n: usize) -> Substitution {
    Substitution::intern(s.interned()[..std::cmp::min(s.len(&Interner), n)].into())
}

/// Return an index of a parameter in the generic type parameter list by it's id.
pub fn param_idx(db: &dyn HirDatabase, id: TypeParamId) -> Option<usize> {
    generics(db.upcast(), id.parent).param_idx(id)
}

pub fn wrap_empty_binders<T>(value: T) -> Binders<T>
where
    T: TypeWalk,
{
    Binders::empty(&Interner, value.shifted_in_from(DebruijnIndex::ONE))
}

pub fn make_only_type_binders<T>(num_vars: usize, value: T) -> Binders<T> {
    Binders::new(
        VariableKinds::from_iter(
            &Interner,
            std::iter::repeat(chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General))
                .take(num_vars),
        ),
        value,
    )
}

impl TraitRef {
    pub fn hir_trait_id(&self) -> TraitId {
        from_chalk_trait_id(self.trait_id)
    }
}

impl<T> Canonical<T> {
    pub fn new(value: T, kinds: impl IntoIterator<Item = TyVariableKind>) -> Self {
        let kinds = kinds.into_iter().map(|tk| {
            chalk_ir::CanonicalVarKind::new(
                chalk_ir::VariableKind::Ty(tk),
                chalk_ir::UniverseIndex::ROOT,
            )
        });
        Self { value, binders: chalk_ir::CanonicalVarKinds::from_iter(&Interner, kinds) }
    }
}

/// A function signature as seen by type inference: Several parameter types and
/// one return type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CallableSig {
    params_and_return: Arc<[Ty]>,
    is_varargs: bool,
}

/// A polymorphic function signature.
pub type PolyFnSig = Binders<CallableSig>;

impl CallableSig {
    pub fn from_params_and_return(mut params: Vec<Ty>, ret: Ty, is_varargs: bool) -> CallableSig {
        params.push(ret);
        CallableSig { params_and_return: params.into(), is_varargs }
    }

    pub fn from_fn_ptr(fn_ptr: &FnPointer) -> CallableSig {
        CallableSig {
            // FIXME: what to do about lifetime params? -> return PolyFnSig
            params_and_return: fn_ptr
                .substitution
                .clone()
                .shifted_out_to(DebruijnIndex::ONE)
                .expect("unexpected lifetime vars in fn ptr")
                .0
                .interned()
                .iter()
                .map(|arg| arg.assert_ty_ref(&Interner).clone())
                .collect(),
            is_varargs: fn_ptr.sig.variadic,
        }
    }

    pub fn params(&self) -> &[Ty] {
        &self.params_and_return[0..self.params_and_return.len() - 1]
    }

    pub fn ret(&self) -> &Ty {
        &self.params_and_return[self.params_and_return.len() - 1]
    }
}

impl Ty {
    pub fn as_reference(&self) -> Option<(&Ty, Lifetime, Mutability)> {
        match self.kind(&Interner) {
            TyKind::Ref(mutability, lifetime, ty) => Some((ty, *lifetime, *mutability)),
            _ => None,
        }
    }

    pub fn as_reference_or_ptr(&self) -> Option<(&Ty, Rawness, Mutability)> {
        match self.kind(&Interner) {
            TyKind::Ref(mutability, _, ty) => Some((ty, Rawness::Ref, *mutability)),
            TyKind::Raw(mutability, ty) => Some((ty, Rawness::RawPtr, *mutability)),
            _ => None,
        }
    }

    pub fn strip_references(&self) -> &Ty {
        let mut t: &Ty = self;

        while let TyKind::Ref(_mutability, _lifetime, ty) = t.kind(&Interner) {
            t = ty;
        }

        t
    }

    pub fn as_adt(&self) -> Option<(hir_def::AdtId, &Substitution)> {
        match self.kind(&Interner) {
            TyKind::Adt(AdtId(adt), parameters) => Some((*adt, parameters)),
            _ => None,
        }
    }

    pub fn as_tuple(&self) -> Option<&Substitution> {
        match self.kind(&Interner) {
            TyKind::Tuple(_, substs) => Some(substs),
            _ => None,
        }
    }

    pub fn as_generic_def(&self, db: &dyn HirDatabase) -> Option<GenericDefId> {
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

    pub fn is_never(&self) -> bool {
        matches!(self.kind(&Interner), TyKind::Never)
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self.kind(&Interner), TyKind::Error)
    }

    pub fn equals_ctor(&self, other: &Ty) -> bool {
        match (self.kind(&Interner), other.kind(&Interner)) {
            (TyKind::Adt(adt, ..), TyKind::Adt(adt2, ..)) => adt == adt2,
            (TyKind::Slice(_), TyKind::Slice(_)) | (TyKind::Array(_), TyKind::Array(_)) => true,
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

    /// If this is a `dyn Trait` type, this returns the `Trait` part.
    fn dyn_trait_ref(&self) -> Option<&TraitRef> {
        match self.kind(&Interner) {
            TyKind::Dyn(dyn_ty) => dyn_ty.bounds.skip_binders().interned().get(0).and_then(|b| {
                match b.skip_binders() {
                    WhereClause::Implemented(trait_ref) => Some(trait_ref),
                    _ => None,
                }
            }),
            _ => None,
        }
    }

    /// If this is a `dyn Trait`, returns that trait.
    pub fn dyn_trait(&self) -> Option<TraitId> {
        self.dyn_trait_ref().map(|it| it.trait_id).map(from_chalk_trait_id)
    }

    fn builtin_deref(&self) -> Option<Ty> {
        match self.kind(&Interner) {
            TyKind::Ref(.., ty) => Some(ty.clone()),
            TyKind::Raw(.., ty) => Some(ty.clone()),
            _ => None,
        }
    }

    pub fn callable_def(&self, db: &dyn HirDatabase) -> Option<CallableDefId> {
        match self.kind(&Interner) {
            &TyKind::FnDef(def, ..) => Some(db.lookup_intern_callable_def(def.into())),
            _ => None,
        }
    }

    pub fn as_fn_def(&self, db: &dyn HirDatabase) -> Option<FunctionId> {
        if let Some(CallableDefId::FunctionId(func)) = self.callable_def(db) {
            Some(func)
        } else {
            None
        }
    }

    pub fn callable_sig(&self, db: &dyn HirDatabase) -> Option<CallableSig> {
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

    /// Returns the type parameters of this type if it has some (i.e. is an ADT
    /// or function); so if `self` is `Option<u32>`, this returns the `u32`.
    pub fn substs(&self) -> Option<&Substitution> {
        match self.kind(&Interner) {
            TyKind::Adt(_, substs)
            | TyKind::FnDef(_, substs)
            | TyKind::Tuple(_, substs)
            | TyKind::OpaqueType(_, substs)
            | TyKind::AssociatedType(_, substs)
            | TyKind::Closure(.., substs) => Some(substs),
            TyKind::Function(FnPointer { substitution: substs, .. }) => Some(&substs.0),
            _ => None,
        }
    }

    fn substs_mut(&mut self) -> Option<&mut Substitution> {
        match self.interned_mut() {
            TyKind::Adt(_, substs)
            | TyKind::FnDef(_, substs)
            | TyKind::Tuple(_, substs)
            | TyKind::OpaqueType(_, substs)
            | TyKind::AssociatedType(_, substs)
            | TyKind::Closure(.., substs) => Some(substs),
            TyKind::Function(FnPointer { substitution: substs, .. }) => Some(&mut substs.0),
            _ => None,
        }
    }

    pub fn impl_trait_bounds(&self, db: &dyn HirDatabase) -> Option<Vec<QuantifiedWhereClause>> {
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
                            .collect_vec();

                        Some(predicates)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub fn associated_type_parent_trait(&self, db: &dyn HirDatabase) -> Option<TraitId> {
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

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum ImplTraitId {
    ReturnTypeImplTrait(hir_def::FunctionId, u16),
    AsyncBlockTypeImplTrait(hir_def::DefWithBodyId, ExprId),
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct ReturnTypeImplTraits {
    pub(crate) impl_traits: Vec<ReturnTypeImplTrait>,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct ReturnTypeImplTrait {
    pub(crate) bounds: Binders<Vec<QuantifiedWhereClause>>,
}

pub fn to_foreign_def_id(id: TypeAliasId) -> ForeignDefId {
    chalk_ir::ForeignDefId(salsa::InternKey::as_intern_id(&id))
}

pub fn from_foreign_def_id(id: ForeignDefId) -> TypeAliasId {
    salsa::InternKey::from_intern_id(id.0)
}

pub fn to_assoc_type_id(id: TypeAliasId) -> AssocTypeId {
    chalk_ir::AssocTypeId(salsa::InternKey::as_intern_id(&id))
}

pub fn from_assoc_type_id(id: AssocTypeId) -> TypeAliasId {
    salsa::InternKey::from_intern_id(id.0)
}

pub fn from_placeholder_idx(db: &dyn HirDatabase, idx: PlaceholderIndex) -> TypeParamId {
    assert_eq!(idx.ui, chalk_ir::UniverseIndex::ROOT);
    let interned_id = salsa::InternKey::from_intern_id(salsa::InternId::from(idx.idx));
    db.lookup_intern_type_param_id(interned_id)
}

pub fn to_placeholder_idx(db: &dyn HirDatabase, id: TypeParamId) -> PlaceholderIndex {
    let interned_id = db.intern_type_param_id(id);
    PlaceholderIndex {
        ui: chalk_ir::UniverseIndex::ROOT,
        idx: salsa::InternKey::as_intern_id(&interned_id).as_usize(),
    }
}

pub fn lt_from_placeholder_idx(db: &dyn HirDatabase, idx: PlaceholderIndex) -> LifetimeParamId {
    assert_eq!(idx.ui, chalk_ir::UniverseIndex::ROOT);
    let interned_id = salsa::InternKey::from_intern_id(salsa::InternId::from(idx.idx));
    db.lookup_intern_lifetime_param_id(interned_id)
}

pub fn to_chalk_trait_id(id: TraitId) -> ChalkTraitId {
    chalk_ir::TraitId(salsa::InternKey::as_intern_id(&id))
}

pub fn from_chalk_trait_id(id: ChalkTraitId) -> TraitId {
    salsa::InternKey::from_intern_id(id.0)
}

pub fn static_lifetime() -> Lifetime {
    LifetimeData::Static.intern(&Interner)
}
