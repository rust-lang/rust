//! `TyBuilder`, a helper for building instances of `Ty` and related types.

use std::iter;

use chalk_ir::{
    cast::{Cast, CastTo, Caster},
    fold::TypeFoldable,
    interner::HasInterner,
    AdtId, BoundVar, DebruijnIndex, Scalar,
};
use hir_def::{
    builtin_type::BuiltinType, generics::TypeOrConstParamData, ConstParamId, GenericDefId, TraitId,
    TypeAliasId,
};
use smallvec::SmallVec;

use crate::{
    consteval::unknown_const_as_generic, db::HirDatabase, infer::unify::InferenceTable, primitive,
    to_assoc_type_id, to_chalk_trait_id, utils::generics, Binders, CallableSig, ConstData,
    ConstValue, GenericArg, GenericArgData, Interner, ProjectionTy, Substitution, TraitRef, Ty,
    TyDefId, TyExt, TyKind, ValueTyDefId,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamKind {
    Type,
    Const(Ty),
}

/// This is a builder for `Ty` or anything that needs a `Substitution`.
pub struct TyBuilder<D> {
    /// The `data` field is used to keep track of what we're building (e.g. an
    /// ADT, a `TraitRef`, ...).
    data: D,
    vec: SmallVec<[GenericArg; 2]>,
    param_kinds: SmallVec<[ParamKind; 2]>,
}

impl<A> TyBuilder<A> {
    fn with_data<B>(self, data: B) -> TyBuilder<B> {
        TyBuilder { data, param_kinds: self.param_kinds, vec: self.vec }
    }
}

impl<D> TyBuilder<D> {
    fn new(data: D, param_kinds: SmallVec<[ParamKind; 2]>) -> TyBuilder<D> {
        TyBuilder { data, vec: SmallVec::with_capacity(param_kinds.len()), param_kinds }
    }

    fn build_internal(self) -> (D, Substitution) {
        assert_eq!(self.vec.len(), self.param_kinds.len());
        for (a, e) in self.vec.iter().zip(self.param_kinds.iter()) {
            self.assert_match_kind(a, e);
        }
        let subst = Substitution::from_iter(Interner, self.vec);
        (self.data, subst)
    }

    pub fn push(mut self, arg: impl CastTo<GenericArg>) -> Self {
        let arg = arg.cast(Interner);
        let expected_kind = &self.param_kinds[self.vec.len()];
        let arg_kind = match arg.data(Interner) {
            chalk_ir::GenericArgData::Ty(_) => ParamKind::Type,
            chalk_ir::GenericArgData::Lifetime(_) => panic!("Got lifetime in TyBuilder::push"),
            chalk_ir::GenericArgData::Const(c) => {
                let c = c.data(Interner);
                ParamKind::Const(c.ty.clone())
            }
        };
        assert_eq!(*expected_kind, arg_kind);
        self.vec.push(arg);
        self
    }

    pub fn remaining(&self) -> usize {
        self.param_kinds.len() - self.vec.len()
    }

    pub fn fill_with_bound_vars(self, debruijn: DebruijnIndex, starting_from: usize) -> Self {
        // self.fill is inlined to make borrow checker happy
        let mut this = self;
        let other = this.param_kinds.iter().skip(this.vec.len());
        let filler = (starting_from..).zip(other).map(|(idx, kind)| match kind {
            ParamKind::Type => {
                GenericArgData::Ty(TyKind::BoundVar(BoundVar::new(debruijn, idx)).intern(Interner))
                    .intern(Interner)
            }
            ParamKind::Const(ty) => GenericArgData::Const(
                ConstData {
                    value: ConstValue::BoundVar(BoundVar::new(debruijn, idx)),
                    ty: ty.clone(),
                }
                .intern(Interner),
            )
            .intern(Interner),
        });
        this.vec.extend(filler.take(this.remaining()).casted(Interner));
        assert_eq!(this.remaining(), 0);
        this
    }

    pub fn fill_with_unknown(self) -> Self {
        // self.fill is inlined to make borrow checker happy
        let mut this = self;
        let filler = this.param_kinds.iter().skip(this.vec.len()).map(|x| match x {
            ParamKind::Type => GenericArgData::Ty(TyKind::Error.intern(Interner)).intern(Interner),
            ParamKind::Const(ty) => unknown_const_as_generic(ty.clone()),
        });
        this.vec.extend(filler.casted(Interner));
        assert_eq!(this.remaining(), 0);
        this
    }

    pub(crate) fn fill_with_inference_vars(self, table: &mut InferenceTable) -> Self {
        self.fill(|x| match x {
            ParamKind::Type => GenericArgData::Ty(table.new_type_var()).intern(Interner),
            ParamKind::Const(ty) => {
                GenericArgData::Const(table.new_const_var(ty.clone())).intern(Interner)
            }
        })
    }

    pub fn fill(mut self, filler: impl FnMut(&ParamKind) -> GenericArg) -> Self {
        self.vec.extend(self.param_kinds.iter().skip(self.vec.len()).map(filler));
        assert_eq!(self.remaining(), 0);
        self
    }

    pub fn use_parent_substs(mut self, parent_substs: &Substitution) -> Self {
        assert!(self.vec.is_empty());
        assert!(parent_substs.len(Interner) <= self.param_kinds.len());
        self.extend(parent_substs.iter(Interner).cloned());
        self
    }

    fn extend(&mut self, it: impl Iterator<Item = GenericArg> + Clone) {
        for x in it.clone().zip(self.param_kinds.iter().skip(self.vec.len())) {
            self.assert_match_kind(&x.0, &x.1);
        }
        self.vec.extend(it);
    }

    fn assert_match_kind(&self, a: &chalk_ir::GenericArg<Interner>, e: &ParamKind) {
        match (a.data(Interner), e) {
            (chalk_ir::GenericArgData::Ty(_), ParamKind::Type)
            | (chalk_ir::GenericArgData::Const(_), ParamKind::Const(_)) => (),
            _ => panic!("Mismatched kinds: {:?}, {:?}, {:?}", a, self.vec, self.param_kinds),
        }
    }
}

impl TyBuilder<()> {
    pub fn unit() -> Ty {
        TyKind::Tuple(0, Substitution::empty(Interner)).intern(Interner)
    }

    pub fn usize() -> Ty {
        TyKind::Scalar(chalk_ir::Scalar::Uint(chalk_ir::UintTy::Usize)).intern(Interner)
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

    pub fn placeholder_subst(db: &dyn HirDatabase, def: impl Into<GenericDefId>) -> Substitution {
        let params = generics(db.upcast(), def.into());
        params.placeholder_subst(db)
    }

    pub fn subst_for_def(db: &dyn HirDatabase, def: impl Into<GenericDefId>) -> TyBuilder<()> {
        let def = def.into();
        let params = generics(db.upcast(), def);
        TyBuilder::new(
            (),
            params
                .iter()
                .map(|(id, data)| match data {
                    TypeOrConstParamData::TypeParamData(_) => ParamKind::Type,
                    TypeOrConstParamData::ConstParamData(_) => {
                        ParamKind::Const(db.const_param_ty(ConstParamId::from_unchecked(id)))
                    }
                })
                .collect(),
        )
    }

    pub fn build(self) -> Substitution {
        let ((), subst) = self.build_internal();
        subst
    }
}

impl TyBuilder<hir_def::AdtId> {
    pub fn adt(db: &dyn HirDatabase, def: hir_def::AdtId) -> TyBuilder<hir_def::AdtId> {
        TyBuilder::subst_for_def(db, def).with_data(def)
    }

    pub fn fill_with_defaults(
        mut self,
        db: &dyn HirDatabase,
        mut fallback: impl FnMut() -> Ty,
    ) -> Self {
        let defaults = db.generic_defaults(self.data.into());
        for default_ty in defaults.iter().skip(self.vec.len()) {
            if let GenericArgData::Ty(x) = default_ty.skip_binders().data(Interner) {
                if x.is_unknown() {
                    self.vec.push(fallback().cast(Interner));
                    continue;
                }
            };
            // each default can depend on the previous parameters
            let subst_so_far = Substitution::from_iter(Interner, self.vec.clone());
            self.vec.push(default_ty.clone().substitute(Interner, &subst_so_far).cast(Interner));
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
        TyBuilder::new(Tuple(size), iter::repeat(ParamKind::Type).take(size).collect())
    }

    pub fn build(self) -> Ty {
        let (Tuple(size), subst) = self.build_internal();
        TyKind::Tuple(size, subst).intern(Interner)
    }
}

impl TyBuilder<TraitId> {
    pub fn trait_ref(db: &dyn HirDatabase, def: TraitId) -> TyBuilder<TraitId> {
        TyBuilder::subst_for_def(db, def).with_data(def)
    }

    pub fn build(self) -> TraitRef {
        let (trait_id, substitution) = self.build_internal();
        TraitRef { trait_id: to_chalk_trait_id(trait_id), substitution }
    }
}

impl TyBuilder<TypeAliasId> {
    pub fn assoc_type_projection(db: &dyn HirDatabase, def: TypeAliasId) -> TyBuilder<TypeAliasId> {
        TyBuilder::subst_for_def(db, def).with_data(def)
    }

    pub fn build(self) -> ProjectionTy {
        let (type_alias, substitution) = self.build_internal();
        ProjectionTy { associated_ty_id: to_assoc_type_id(type_alias), substitution }
    }
}

impl<T: HasInterner<Interner = Interner> + TypeFoldable<Interner>> TyBuilder<Binders<T>> {
    fn subst_binders(b: Binders<T>) -> Self {
        let param_kinds = b
            .binders
            .iter(Interner)
            .map(|x| match x {
                chalk_ir::VariableKind::Ty(_) => ParamKind::Type,
                chalk_ir::VariableKind::Lifetime => panic!("Got lifetime parameter"),
                chalk_ir::VariableKind::Const(ty) => ParamKind::Const(ty.clone()),
            })
            .collect();
        TyBuilder::new(b, param_kinds)
    }

    pub fn build(self) -> T {
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
