//! `TyBuilder`, a helper for building instances of `Ty` and related types.

use chalk_ir::{
    DebruijnIndex,
    cast::{Cast, Caster},
};
use hir_def::{GenericDefId, GenericParamId, TraitId};
use smallvec::SmallVec;

use crate::{
    BoundVar, GenericArg, GenericArgData, Interner, Substitution, TraitRef, Ty, TyKind,
    consteval::unknown_const_as_generic,
    db::HirDatabase,
    error_lifetime,
    generics::generics,
    infer::unify::InferenceTable,
    next_solver::{
        DbInterner, EarlyBinder,
        mapping::{ChalkToNextSolver, NextSolverToChalk},
    },
    to_chalk_trait_id,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParamKind {
    Type,
    Lifetime,
    Const(Ty),
}

/// This is a builder for `Ty` or anything that needs a `Substitution`.
pub(crate) struct TyBuilder<D> {
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

    fn build_internal(self) -> (D, Substitution) {
        assert_eq!(
            self.vec.len(),
            self.param_kinds.len(),
            "{} args received, {} expected ({:?})",
            self.vec.len(),
            self.param_kinds.len(),
            &self.param_kinds
        );
        for (a, e) in self.vec.iter().zip(self.param_kinds.iter()) {
            self.assert_match_kind(a, e);
        }
        let subst = Substitution::from_iter(
            Interner,
            self.parent_subst.iter(Interner).cloned().chain(self.vec),
        );
        (self.data, subst)
    }

    pub(crate) fn remaining(&self) -> usize {
        self.param_kinds.len() - self.vec.len()
    }

    pub(crate) fn fill_with_bound_vars(
        self,
        debruijn: DebruijnIndex,
        starting_from: usize,
    ) -> Self {
        // self.fill is inlined to make borrow checker happy
        let mut this = self;
        let other = &this.param_kinds[this.vec.len()..];
        let filler = (starting_from..).zip(other).map(|(idx, kind)| match kind {
            ParamKind::Type => BoundVar::new(debruijn, idx).to_ty(Interner).cast(Interner),
            ParamKind::Const(ty) => {
                BoundVar::new(debruijn, idx).to_const(Interner, ty.clone()).cast(Interner)
            }
            ParamKind::Lifetime => {
                BoundVar::new(debruijn, idx).to_lifetime(Interner).cast(Interner)
            }
        });
        this.vec.extend(filler.take(this.remaining()).casted(Interner));
        assert_eq!(this.remaining(), 0);
        this
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn fill_with_inference_vars(self, table: &mut InferenceTable<'_>) -> Self {
        self.fill(|x| {
            match x {
                ParamKind::Type => crate::next_solver::GenericArg::Ty(table.next_ty_var()),
                ParamKind::Const(_) => table.next_const_var().into(),
                ParamKind::Lifetime => table.next_region_var().into(),
            }
            .to_chalk(table.interner())
        })
    }

    pub(crate) fn fill(mut self, filler: impl FnMut(&ParamKind) -> GenericArg) -> Self {
        self.vec.extend(self.param_kinds[self.vec.len()..].iter().map(filler));
        assert_eq!(self.remaining(), 0);
        self
    }

    fn assert_match_kind(&self, a: &chalk_ir::GenericArg<Interner>, e: &ParamKind) {
        match (a.data(Interner), e) {
            (GenericArgData::Ty(_), ParamKind::Type)
            | (GenericArgData::Const(_), ParamKind::Const(_))
            | (GenericArgData::Lifetime(_), ParamKind::Lifetime) => (),
            _ => panic!("Mismatched kinds: {a:?}, {:?}, {:?}", self.vec, self.param_kinds),
        }
    }
}

impl TyBuilder<()> {
    pub(crate) fn usize() -> Ty {
        TyKind::Scalar(chalk_ir::Scalar::Uint(chalk_ir::UintTy::Usize)).intern(Interner)
    }

    pub(crate) fn unknown_subst(
        db: &dyn HirDatabase,
        def: impl Into<GenericDefId>,
    ) -> Substitution {
        let interner = DbInterner::conjure();
        let params = generics(db, def.into());
        Substitution::from_iter(
            Interner,
            params.iter_id().map(|id| match id {
                GenericParamId::TypeParamId(_) => TyKind::Error.intern(Interner).cast(Interner),
                GenericParamId::ConstParamId(id) => {
                    unknown_const_as_generic(db.const_param_ty_ns(id))
                        .to_chalk(interner)
                        .cast(Interner)
                }
                GenericParamId::LifetimeParamId(_) => error_lifetime().cast(Interner),
            }),
        )
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn subst_for_def(
        db: &dyn HirDatabase,
        def: impl Into<GenericDefId>,
        parent_subst: Option<Substitution>,
    ) -> TyBuilder<()> {
        let generics = generics(db, def.into());
        assert!(generics.parent_generics().is_some() == parent_subst.is_some());
        let params = generics
            .iter_self()
            .map(|(id, _data)| match id {
                GenericParamId::TypeParamId(_) => ParamKind::Type,
                GenericParamId::ConstParamId(id) => ParamKind::Const(db.const_param_ty(id)),
                GenericParamId::LifetimeParamId(_) => ParamKind::Lifetime,
            })
            .collect();
        TyBuilder::new((), params, parent_subst)
    }

    pub(crate) fn build(self) -> Substitution {
        let ((), subst) = self.build_internal();
        subst
    }
}

impl TyBuilder<TraitId> {
    pub(crate) fn trait_ref(db: &dyn HirDatabase, def: TraitId) -> TyBuilder<TraitId> {
        TyBuilder::subst_for_def(db, def, None).with_data(def)
    }

    pub(crate) fn build(self) -> TraitRef {
        let (trait_id, substitution) = self.build_internal();
        TraitRef { trait_id: to_chalk_trait_id(trait_id), substitution }
    }
}

impl<'db, T: rustc_type_ir::TypeFoldable<DbInterner<'db>>> TyBuilder<EarlyBinder<'db, T>> {
    pub(crate) fn build(self, interner: DbInterner<'db>) -> T {
        let (b, subst) = self.build_internal();
        let args: crate::next_solver::GenericArgs<'db> = subst.to_nextsolver(interner);
        b.instantiate(interner, args)
    }
}

impl<'db> TyBuilder<EarlyBinder<'db, crate::next_solver::Ty<'db>>> {
    pub(crate) fn impl_self_ty(
        db: &'db dyn HirDatabase,
        def: hir_def::ImplId,
    ) -> TyBuilder<EarlyBinder<'db, crate::next_solver::Ty<'db>>> {
        TyBuilder::subst_for_def(db, def, None).with_data(db.impl_self_ty(def))
    }
}
