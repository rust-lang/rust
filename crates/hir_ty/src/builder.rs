//! `TyBuilder`, a helper for building instances of `Ty` and related types.

use std::iter;

use chalk_ir::{
    cast::{Cast, CastTo, Caster},
    fold::Fold,
    interner::HasInterner,
    AdtId, BoundVar, DebruijnIndex, Scalar,
};
use hir_def::{builtin_type::BuiltinType, GenericDefId, TraitId, TypeAliasId};
use smallvec::SmallVec;

use crate::{
    db::HirDatabase, primitive, to_assoc_type_id, to_chalk_trait_id, utils::generics, Binders,
    CallableSig, GenericArg, Interner, ProjectionTy, Substitution, TraitRef, Ty, TyDefId, TyExt,
    TyKind, ValueTyDefId,
};

/// This is a builder for `Ty` or anything that needs a `Substitution`.
pub struct TyBuilder<D> {
    /// The `data` field is used to keep track of what we're building (e.g. an
    /// ADT, a `TraitRef`, ...).
    data: D,
    vec: SmallVec<[GenericArg; 2]>,
    param_count: usize,
}

impl<D> TyBuilder<D> {
    fn new(data: D, param_count: usize) -> TyBuilder<D> {
        TyBuilder { data, param_count, vec: SmallVec::with_capacity(param_count) }
    }

    fn build_internal(self) -> (D, Substitution) {
        assert_eq!(self.vec.len(), self.param_count);
        let subst = Substitution::from_iter(Interner, self.vec);
        (self.data, subst)
    }

    pub fn push(mut self, arg: impl CastTo<GenericArg>) -> Self {
        self.vec.push(arg.cast(Interner));
        self
    }

    pub fn remaining(&self) -> usize {
        self.param_count - self.vec.len()
    }

    pub fn fill_with_bound_vars(self, debruijn: DebruijnIndex, starting_from: usize) -> Self {
        self.fill(
            (starting_from..)
                .map(|idx| TyKind::BoundVar(BoundVar::new(debruijn, idx)).intern(Interner)),
        )
    }

    pub fn fill_with_unknown(self) -> Self {
        self.fill(iter::repeat(TyKind::Error.intern(Interner)))
    }

    pub fn fill(mut self, filler: impl Iterator<Item = impl CastTo<GenericArg>>) -> Self {
        self.vec.extend(filler.take(self.remaining()).casted(Interner));
        assert_eq!(self.remaining(), 0);
        self
    }

    pub fn use_parent_substs(mut self, parent_substs: &Substitution) -> Self {
        assert!(self.vec.is_empty());
        assert!(parent_substs.len(Interner) <= self.param_count);
        self.vec.extend(parent_substs.iter(Interner).cloned());
        self
    }
}

impl TyBuilder<()> {
    pub fn unit() -> Ty {
        TyKind::Tuple(0, Substitution::empty(Interner)).intern(Interner)
    }

    pub fn fn_ptr(sig: CallableSig) -> Ty {
        TyKind::Function(sig.to_fn_ptr()).intern(Interner)
    }

    pub fn builtin(builtin: BuiltinType) -> Ty {
        match builtin {
            BuiltinType::Char => TyKind::Scalar(Scalar::Char).intern(Interner),
            BuiltinType::Bool => TyKind::Scalar(Scalar::Bool).intern(Interner),
            BuiltinType::Str => TyKind::Str.intern(Interner),
            BuiltinType::Int(t) => {
                TyKind::Scalar(Scalar::Int(primitive::int_ty_from_builtin(t))).intern(Interner)
            }
            BuiltinType::Uint(t) => {
                TyKind::Scalar(Scalar::Uint(primitive::uint_ty_from_builtin(t))).intern(Interner)
            }
            BuiltinType::Float(t) => {
                TyKind::Scalar(Scalar::Float(primitive::float_ty_from_builtin(t))).intern(Interner)
            }
        }
    }

    pub fn slice(argument: Ty) -> Ty {
        TyKind::Slice(argument).intern(Interner)
    }

    pub fn type_params_subst(db: &dyn HirDatabase, def: impl Into<GenericDefId>) -> Substitution {
        let params = generics(db.upcast(), def.into());
        params.type_params_subst(db)
    }

    pub fn subst_for_def(db: &dyn HirDatabase, def: impl Into<GenericDefId>) -> TyBuilder<()> {
        let def = def.into();
        let params = generics(db.upcast(), def);
        let param_count = params.len();
        TyBuilder::new((), param_count)
    }

    pub fn build(self) -> Substitution {
        let ((), subst) = self.build_internal();
        subst
    }
}

impl TyBuilder<hir_def::AdtId> {
    pub fn adt(db: &dyn HirDatabase, adt: hir_def::AdtId) -> TyBuilder<hir_def::AdtId> {
        let generics = generics(db.upcast(), adt.into());
        let param_count = generics.len();
        TyBuilder::new(adt, param_count)
    }

    pub fn fill_with_defaults(
        mut self,
        db: &dyn HirDatabase,
        mut fallback: impl FnMut() -> Ty,
    ) -> Self {
        let defaults = db.generic_defaults(self.data.into());
        for default_ty in defaults.iter().skip(self.vec.len()) {
            if default_ty.skip_binders().is_unknown() {
                self.vec.push(fallback().cast(Interner));
            } else {
                // each default can depend on the previous parameters
                let subst_so_far = Substitution::from_iter(Interner, self.vec.clone());
                self.vec
                    .push(default_ty.clone().substitute(Interner, &subst_so_far).cast(Interner));
            }
        }
        self
    }

    pub fn build(self) -> Ty {
        let (adt, subst) = self.build_internal();
        TyKind::Adt(AdtId(adt), subst).intern(Interner)
    }
}

pub struct Tuple(usize);
impl TyBuilder<Tuple> {
    pub fn tuple(size: usize) -> TyBuilder<Tuple> {
        TyBuilder::new(Tuple(size), size)
    }

    pub fn build(self) -> Ty {
        let (Tuple(size), subst) = self.build_internal();
        TyKind::Tuple(size, subst).intern(Interner)
    }
}

impl TyBuilder<TraitId> {
    pub fn trait_ref(db: &dyn HirDatabase, trait_id: TraitId) -> TyBuilder<TraitId> {
        let generics = generics(db.upcast(), trait_id.into());
        let param_count = generics.len();
        TyBuilder::new(trait_id, param_count)
    }

    pub fn build(self) -> TraitRef {
        let (trait_id, substitution) = self.build_internal();
        TraitRef { trait_id: to_chalk_trait_id(trait_id), substitution }
    }
}

impl TyBuilder<TypeAliasId> {
    pub fn assoc_type_projection(
        db: &dyn HirDatabase,
        type_alias: TypeAliasId,
    ) -> TyBuilder<TypeAliasId> {
        let generics = generics(db.upcast(), type_alias.into());
        let param_count = generics.len();
        TyBuilder::new(type_alias, param_count)
    }

    pub fn build(self) -> ProjectionTy {
        let (type_alias, substitution) = self.build_internal();
        ProjectionTy { associated_ty_id: to_assoc_type_id(type_alias), substitution }
    }
}

impl<T: HasInterner<Interner = Interner> + Fold<Interner>> TyBuilder<Binders<T>> {
    fn subst_binders(b: Binders<T>) -> Self {
        let param_count = b.binders.len(Interner);
        TyBuilder::new(b, param_count)
    }

    pub fn build(self) -> <T as Fold<Interner>>::Result {
        let (b, subst) = self.build_internal();
        b.substitute(Interner, &subst)
    }
}

impl TyBuilder<Binders<Ty>> {
    pub fn def_ty(db: &dyn HirDatabase, def: TyDefId) -> TyBuilder<Binders<Ty>> {
        TyBuilder::subst_binders(db.ty(def))
    }

    pub fn impl_self_ty(db: &dyn HirDatabase, def: hir_def::ImplId) -> TyBuilder<Binders<Ty>> {
        TyBuilder::subst_binders(db.impl_self_ty(def))
    }

    pub fn value_ty(db: &dyn HirDatabase, def: ValueTyDefId) -> TyBuilder<Binders<Ty>> {
        TyBuilder::subst_binders(db.value_ty(def))
    }
}
