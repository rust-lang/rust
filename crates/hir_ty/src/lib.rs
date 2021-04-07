//! The type system. We currently use this to infer types for completion, hover
//! information and various assists.

#![allow(unused)]

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

pub mod display;
pub mod db;
pub mod diagnostics;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod test_db;

use std::sync::Arc;

use base_db::salsa;
use chalk_ir::{
    cast::{CastTo, Caster},
    fold::{Fold, Shift},
    interner::HasInterner,
    UintTy,
};
use hir_def::{
    expr::ExprId, type_ref::Rawness, ConstParamId, LifetimeParamId, TraitId, TypeAliasId,
    TypeParamId,
};

use crate::{db::HirDatabase, display::HirDisplay, utils::generics};

pub use autoderef::autoderef;
pub use builder::TyBuilder;
pub use chalk_ext::*;
pub use infer::{could_unify, InferenceResult};
pub use lower::{
    associated_type_shorthand_candidates, callable_item_sig, CallableDefId, ImplTraitLoweringMode,
    TyDefId, TyLoweringContext, ValueTyDefId,
};
pub use traits::{chalk::Interner, TraitEnvironment};
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
pub type Binders<T> = chalk_ir::Binders<T>;
pub type Substitution = chalk_ir::Substitution<Interner>;
pub type GenericArg = chalk_ir::GenericArg<Interner>;
pub type GenericArgData = chalk_ir::GenericArgData<Interner>;

pub type Ty = chalk_ir::Ty<Interner>;
pub type TyKind = chalk_ir::TyKind<Interner>;
pub type DynTy = chalk_ir::DynTy<Interner>;
pub type FnPointer = chalk_ir::FnPointer<Interner>;
// pub type FnSubst = chalk_ir::FnSubst<Interner>;
pub use chalk_ir::FnSubst;
pub type ProjectionTy = chalk_ir::ProjectionTy<Interner>;
pub type AliasTy = chalk_ir::AliasTy<Interner>;
pub type OpaqueTy = chalk_ir::OpaqueTy<Interner>;
pub type InferenceVar = chalk_ir::InferenceVar;

pub type Lifetime = chalk_ir::Lifetime<Interner>;
pub type LifetimeData = chalk_ir::LifetimeData<Interner>;
pub type LifetimeOutlives = chalk_ir::LifetimeOutlives<Interner>;

pub type Const = chalk_ir::Const<Interner>;
pub type ConstData = chalk_ir::ConstData<Interner>;
pub type ConstValue = chalk_ir::ConstValue<Interner>;
pub type ConcreteConst = chalk_ir::ConcreteConst<Interner>;

pub type ChalkTraitId = chalk_ir::TraitId<Interner>;

pub type FnSig = chalk_ir::FnSig<Interner>;

pub type InEnvironment<T> = chalk_ir::InEnvironment<T>;
pub type DomainGoal = chalk_ir::DomainGoal<Interner>;
pub type AliasEq = chalk_ir::AliasEq<Interner>;
pub type Solution = chalk_solve::Solution<Interner>;
pub type ConstrainedSubst = chalk_ir::ConstrainedSubst<Interner>;
pub type Guidance = chalk_solve::Guidance<Interner>;
pub type WhereClause = chalk_ir::WhereClause<Interner>;

// FIXME: get rid of this
pub fn subst_prefix(s: &Substitution, n: usize) -> Substitution {
    Substitution::from_iter(
        &Interner,
        s.interned()[..std::cmp::min(s.len(&Interner), n)].iter().cloned(),
    )
}

/// Return an index of a parameter in the generic type parameter list by it's id.
pub fn param_idx(db: &dyn HirDatabase, id: TypeParamId) -> Option<usize> {
    generics(db.upcast(), id.parent).param_idx(id)
}

pub fn wrap_empty_binders<T>(value: T) -> Binders<T>
where
    T: Fold<Interner, Result = T> + HasInterner<Interner = Interner>,
{
    Binders::empty(&Interner, value.shifted_in_from(&Interner, DebruijnIndex::ONE))
}

pub fn make_only_type_binders<T: HasInterner<Interner = Interner>>(
    num_vars: usize,
    value: T,
) -> Binders<T> {
    Binders::new(
        VariableKinds::from_iter(
            &Interner,
            std::iter::repeat(chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General))
                .take(num_vars),
        ),
        value,
    )
}

// FIXME: get rid of this
pub fn make_canonical<T: HasInterner<Interner = Interner>>(
    value: T,
    kinds: impl IntoIterator<Item = TyVariableKind>,
) -> Canonical<T> {
    let kinds = kinds.into_iter().map(|tk| {
        chalk_ir::CanonicalVarKind::new(
            chalk_ir::VariableKind::Ty(tk),
            chalk_ir::UniverseIndex::ROOT,
        )
    });
    Canonical { value, binders: chalk_ir::CanonicalVarKinds::from_iter(&Interner, kinds) }
}

pub type TraitRef = chalk_ir::TraitRef<Interner>;

pub type QuantifiedWhereClause = Binders<WhereClause>;

pub type QuantifiedWhereClauses = chalk_ir::QuantifiedWhereClauses<Interner>;

pub type Canonical<T> = chalk_ir::Canonical<T>;

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
                .shifted_out_to(&Interner, DebruijnIndex::ONE)
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

impl Fold<Interner> for CallableSig {
    type Result = CallableSig;

    fn fold_with<'i>(
        self,
        folder: &mut dyn chalk_ir::fold::Folder<'i, Interner>,
        outer_binder: DebruijnIndex,
    ) -> chalk_ir::Fallible<Self::Result>
    where
        Interner: 'i,
    {
        let vec = self.params_and_return.to_vec();
        let folded = vec.fold_with(folder, outer_binder)?;
        Ok(CallableSig { params_and_return: folded.into(), is_varargs: self.is_varargs })
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

pub fn const_from_placeholder_idx(db: &dyn HirDatabase, idx: PlaceholderIndex) -> ConstParamId {
    assert_eq!(idx.ui, chalk_ir::UniverseIndex::ROOT);
    let interned_id = salsa::InternKey::from_intern_id(salsa::InternId::from(idx.idx));
    db.lookup_intern_const_param_id(interned_id)
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

pub fn dummy_usize_const() -> Const {
    let usize_ty = chalk_ir::TyKind::Scalar(Scalar::Uint(UintTy::Usize)).intern(&Interner);
    chalk_ir::ConstData {
        ty: usize_ty,
        value: chalk_ir::ConstValue::Concrete(chalk_ir::ConcreteConst { interned: () }),
    }
    .intern(&Interner)
}

pub(crate) fn fold_free_vars<T: HasInterner<Interner = Interner> + Fold<Interner>>(
    t: T,
    f: impl FnMut(BoundVar, DebruijnIndex) -> Ty,
) -> T::Result {
    use chalk_ir::{fold::Folder, Fallible};
    struct FreeVarFolder<F>(F);
    impl<'i, F: FnMut(BoundVar, DebruijnIndex) -> Ty + 'i> Folder<'i, Interner> for FreeVarFolder<F> {
        fn as_dyn(&mut self) -> &mut dyn Folder<'i, Interner> {
            self
        }

        fn interner(&self) -> &'i Interner {
            &Interner
        }

        fn fold_free_var_ty(
            &mut self,
            bound_var: BoundVar,
            outer_binder: DebruijnIndex,
        ) -> Fallible<Ty> {
            Ok(self.0(bound_var, outer_binder))
        }
    }
    t.fold_with(&mut FreeVarFolder(f), DebruijnIndex::INNERMOST).expect("fold failed unexpectedly")
}
