//! `TyBuilder`, a helper for building instances of `Ty` and related types.

use std::iter;

use chalk_ir::{
    cast::{Cast, CastTo, Caster},
    fold::TypeFoldable,
    interner::HasInterner,
    AdtId, DebruijnIndex, Scalar,
};
use hir_def::{
    builtin_type::BuiltinType, generics::TypeOrConstParamData, ConstParamId, DefWithBodyId,
    GenericDefId, TraitId, TypeAliasId,
};
use smallvec::SmallVec;

use crate::{
    consteval::unknown_const_as_generic, db::HirDatabase, infer::unify::InferenceTable, primitive,
    to_assoc_type_id, to_chalk_trait_id, utils::generics, Binders, BoundVar, CallableSig,
    GenericArg, Interner, ProjectionTy, Substitution, TraitRef, Ty, TyDefId, TyExt, TyKind,
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
    parent_subst: Substitution,
}

impl<A> TyBuilder<A> {
    fn with_data<B>(self, data: B) -> TyBuilder<B> {
        TyBuilder {
            data,
            vec: self.vec,
            param_kinds: self.param_kinds,
            parent_subst: self.parent_subst,
        }
    }
}

impl<D> TyBuilder<D> {
    fn new(
        data: D,
        param_kinds: SmallVec<[ParamKind; 2]>,
        parent_subst: Option<Substitution>,
    ) -> Self {
        let parent_subst = parent_subst.unwrap_or_else(|| Substitution::empty(Interner));
        Self { data, vec: SmallVec::with_capacity(param_kinds.len()), param_kinds, parent_subst }
    }

    fn new_empty(data: D) -> Self {
        TyBuilder::new(data, SmallVec::new(), None)
    }

    fn build_internal(self) -> (D, Substitution) {
        assert_eq!(self.vec.len(), self.param_kinds.len(), "{:?}", &self.param_kinds);
        for (a, e) in self.vec.iter().zip(self.param_kinds.iter()) {
            self.assert_match_kind(a, e);
        }
        let subst = Substitution::from_iter(
            Interner,
            self.vec.into_iter().chain(self.parent_subst.iter(Interner).cloned()),
        );
        (self.data, subst)
    }

    pub fn push(mut self, arg: impl CastTo<GenericArg>) -> Self {
        assert!(self.remaining() > 0);
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
        let other = &this.param_kinds[this.vec.len()..];
        let filler = (starting_from..).zip(other).map(|(idx, kind)| match kind {
            ParamKind::Type => BoundVar::new(debruijn, idx).to_ty(Interner).cast(Interner),
            ParamKind::Const(ty) => {
                BoundVar::new(debruijn, idx).to_const(Interner, ty.clone()).cast(Interner)
            }
        });
        this.vec.extend(filler.take(this.remaining()).casted(Interner));
        assert_eq!(this.remaining(), 0);
        this
    }

    pub fn fill_with_unknown(self) -> Self {
        // self.fill is inlined to make borrow checker happy
        let mut this = self;
        let filler = this.param_kinds[this.vec.len()..].iter().map(|x| match x {
            ParamKind::Type => TyKind::Error.intern(Interner).cast(Interner),
            ParamKind::Const(ty) => unknown_const_as_generic(ty.clone()),
        });
        this.vec.extend(filler.casted(Interner));
        assert_eq!(this.remaining(), 0);
        this
    }

    pub(crate) fn fill_with_inference_vars(self, table: &mut InferenceTable<'_>) -> Self {
        self.fill(|x| match x {
            ParamKind::Type => table.new_type_var().cast(Interner),
            ParamKind::Const(ty) => table.new_const_var(ty.clone()).cast(Interner),
        })
    }

    pub fn fill(mut self, filler: impl FnMut(&ParamKind) -> GenericArg) -> Self {
        self.vec.extend(self.param_kinds[self.vec.len()..].iter().map(filler));
        assert_eq!(self.remaining(), 0);
        self
    }

    fn assert_match_kind(&self, a: &chalk_ir::GenericArg<Interner>, e: &ParamKind) {
        match (a.data(Interner), e) {
            (chalk_ir::GenericArgData::Ty(_), ParamKind::Type)
            | (chalk_ir::GenericArgData::Const(_), ParamKind::Const(_)) => (),
            _ => panic!("Mismatched kinds: {a:?}, {:?}, {:?}", self.vec, self.param_kinds),
        }
    }
}

impl TyBuilder<()> {
    pub fn unit() -> Ty {
        TyKind::Tuple(0, Substitution::empty(Interner)).intern(Interner)
    }

    // FIXME: rustc's ty is dependent on the adt type, maybe we need to do that as well
    pub fn discr_ty() -> Ty {
        TyKind::Scalar(chalk_ir::Scalar::Int(chalk_ir::IntTy::I128)).intern(Interner)
    }

    pub fn bool() -> Ty {
        TyKind::Scalar(chalk_ir::Scalar::Bool).intern(Interner)
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

    pub fn unknown_subst(db: &dyn HirDatabase, def: impl Into<GenericDefId>) -> Substitution {
        let params = generics(db.upcast(), def.into());
        Substitution::from_iter(
            Interner,
            params.iter_id().map(|id| match id {
                either::Either::Left(_) => TyKind::Error.intern(Interner).cast(Interner),
                either::Either::Right(id) => {
                    unknown_const_as_generic(db.const_param_ty(id)).cast(Interner)
                }
            }),
        )
    }

    pub fn subst_for_def(
        db: &dyn HirDatabase,
        def: impl Into<GenericDefId>,
        parent_subst: Option<Substitution>,
    ) -> TyBuilder<()> {
        let generics = generics(db.upcast(), def.into());
        assert!(generics.parent_generics().is_some() == parent_subst.is_some());
        let params = generics
            .iter_self()
            .map(|(id, data)| match data {
                TypeOrConstParamData::TypeParamData(_) => ParamKind::Type,
                TypeOrConstParamData::ConstParamData(_) => {
                    ParamKind::Const(db.const_param_ty(ConstParamId::from_unchecked(id)))
                }
            })
            .collect();
        TyBuilder::new((), params, parent_subst)
    }

    /// Creates a `TyBuilder` to build `Substitution` for a generator defined in `parent`.
    ///
    /// A generator's substitution consists of:
    /// - resume type of generator
    /// - yield type of generator ([`Generator::Yield`](std::ops::Generator::Yield))
    /// - return type of generator ([`Generator::Return`](std::ops::Generator::Return))
    /// - generic parameters in scope on `parent`
    /// in this order.
    ///
    /// This method prepopulates the builder with placeholder substitution of `parent`, so you
    /// should only push exactly 3 `GenericArg`s before building.
    pub fn subst_for_generator(db: &dyn HirDatabase, parent: DefWithBodyId) -> TyBuilder<()> {
        let parent_subst =
            parent.as_generic_def_id().map(|p| generics(db.upcast(), p).placeholder_subst(db));
        // These represent resume type, yield type, and return type of generator.
        let params = std::iter::repeat(ParamKind::Type).take(3).collect();
        TyBuilder::new((), params, parent_subst)
    }

    pub fn subst_for_closure(
        db: &dyn HirDatabase,
        parent: DefWithBodyId,
        sig_ty: Ty,
    ) -> Substitution {
        let sig_ty = sig_ty.cast(Interner);
        let self_subst = iter::once(&sig_ty);
        let Some(parent) = parent.as_generic_def_id() else {
            return Substitution::from_iter(Interner, self_subst);
        };
        Substitution::from_iter(
            Interner,
            self_subst
                .chain(generics(db.upcast(), parent).placeholder_subst(db).iter(Interner))
                .cloned()
                .collect::<Vec<_>>(),
        )
    }

    pub fn build(self) -> Substitution {
        let ((), subst) = self.build_internal();
        subst
    }
}

impl TyBuilder<hir_def::AdtId> {
    pub fn adt(db: &dyn HirDatabase, def: hir_def::AdtId) -> TyBuilder<hir_def::AdtId> {
        TyBuilder::subst_for_def(db, def, None).with_data(def)
    }

    pub fn fill_with_defaults(
        mut self,
        db: &dyn HirDatabase,
        mut fallback: impl FnMut() -> Ty,
    ) -> Self {
        // Note that we're building ADT, so we never have parent generic parameters.
        let defaults = db.generic_defaults(self.data.into());
        let dummy_ty = TyKind::Error.intern(Interner).cast(Interner);
        for default_ty in defaults.iter().skip(self.vec.len()) {
            // NOTE(skip_binders): we only check if the arg type is error type.
            if let Some(x) = default_ty.skip_binders().ty(Interner) {
                if x.is_unknown() {
                    self.vec.push(fallback().cast(Interner));
                    continue;
                }
            }
            // Each default can only depend on the previous parameters.
            // FIXME: we don't handle const generics here.
            let subst_so_far = Substitution::from_iter(
                Interner,
                self.vec
                    .iter()
                    .cloned()
                    .chain(iter::repeat(dummy_ty.clone()))
                    .take(self.param_kinds.len()),
            );
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
        TyBuilder::new(Tuple(size), iter::repeat(ParamKind::Type).take(size).collect(), None)
    }

    pub fn build(self) -> Ty {
        let (Tuple(size), subst) = self.build_internal();
        TyKind::Tuple(size, subst).intern(Interner)
    }

    pub fn tuple_with<I>(elements: I) -> Ty
    where
        I: IntoIterator<Item = Ty>,
        <I as IntoIterator>::IntoIter: ExactSizeIterator,
    {
        let elements = elements.into_iter();
        let len = elements.len();
        let mut b =
            TyBuilder::new(Tuple(len), iter::repeat(ParamKind::Type).take(len).collect(), None);
        for e in elements {
            b = b.push(e);
        }
        b.build()
    }
}

impl TyBuilder<TraitId> {
    pub fn trait_ref(db: &dyn HirDatabase, def: TraitId) -> TyBuilder<TraitId> {
        TyBuilder::subst_for_def(db, def, None).with_data(def)
    }

    pub fn build(self) -> TraitRef {
        let (trait_id, substitution) = self.build_internal();
        TraitRef { trait_id: to_chalk_trait_id(trait_id), substitution }
    }
}

impl TyBuilder<TypeAliasId> {
    pub fn assoc_type_projection(
        db: &dyn HirDatabase,
        def: TypeAliasId,
        parent_subst: Option<Substitution>,
    ) -> TyBuilder<TypeAliasId> {
        TyBuilder::subst_for_def(db, def, parent_subst).with_data(def)
    }

    pub fn build(self) -> ProjectionTy {
        let (type_alias, substitution) = self.build_internal();
        ProjectionTy { associated_ty_id: to_assoc_type_id(type_alias), substitution }
    }
}

impl<T: HasInterner<Interner = Interner> + TypeFoldable<Interner>> TyBuilder<Binders<T>> {
    pub fn build(self) -> T {
        let (b, subst) = self.build_internal();
        b.substitute(Interner, &subst)
    }
}

impl TyBuilder<Binders<Ty>> {
    pub fn def_ty(
        db: &dyn HirDatabase,
        def: TyDefId,
        parent_subst: Option<Substitution>,
    ) -> TyBuilder<Binders<Ty>> {
        let poly_ty = db.ty(def);
        let id: GenericDefId = match def {
            TyDefId::BuiltinType(_) => {
                assert!(parent_subst.is_none());
                return TyBuilder::new_empty(poly_ty);
            }
            TyDefId::AdtId(id) => id.into(),
            TyDefId::TypeAliasId(id) => id.into(),
        };
        TyBuilder::subst_for_def(db, id, parent_subst).with_data(poly_ty)
    }

    pub fn impl_self_ty(db: &dyn HirDatabase, def: hir_def::ImplId) -> TyBuilder<Binders<Ty>> {
        TyBuilder::subst_for_def(db, def, None).with_data(db.impl_self_ty(def))
    }
}
