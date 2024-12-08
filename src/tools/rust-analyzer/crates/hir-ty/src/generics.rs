//! Utilities for working with generics.
//!
//! The layout for generics as expected by chalk are as follows:
//! - Optional Self parameter
//! - Lifetime parameters
//! - Type or Const parameters
//! - Parent parameters
//!
//! where parent follows the same scheme.
use std::ops;

use chalk_ir::{cast::Cast as _, BoundVar, DebruijnIndex};
use hir_def::{
    db::DefDatabase,
    generics::{
        GenericParamDataRef, GenericParams, LifetimeParamData, TypeOrConstParamData,
        TypeParamProvenance,
    },
    type_ref::TypesMap,
    ConstParamId, GenericDefId, GenericParamId, ItemContainerId, LifetimeParamId,
    LocalLifetimeParamId, LocalTypeOrConstParamId, Lookup, TypeOrConstParamId, TypeParamId,
};
use itertools::chain;
use stdx::TupleExt;
use triomphe::Arc;

use crate::{db::HirDatabase, lt_to_placeholder_idx, to_placeholder_idx, Interner, Substitution};

pub(crate) fn generics(db: &dyn DefDatabase, def: GenericDefId) -> Generics {
    let parent_generics = parent_generic_def(db, def).map(|def| Box::new(generics(db, def)));
    let params = db.generic_params(def);
    let has_trait_self_param = params.trait_self_param().is_some();
    Generics { def, params, parent_generics, has_trait_self_param }
}
#[derive(Clone, Debug)]
pub(crate) struct Generics {
    def: GenericDefId,
    params: Arc<GenericParams>,
    parent_generics: Option<Box<Generics>>,
    has_trait_self_param: bool,
}

impl<T> ops::Index<T> for Generics
where
    GenericParams: ops::Index<T>,
{
    type Output = <GenericParams as ops::Index<T>>::Output;
    fn index(&self, index: T) -> &Self::Output {
        &self.params[index]
    }
}

impl Generics {
    pub(crate) fn def(&self) -> GenericDefId {
        self.def
    }

    pub(crate) fn iter_id(&self) -> impl Iterator<Item = GenericParamId> + '_ {
        self.iter_self_id().chain(self.iter_parent_id())
    }

    pub(crate) fn iter_self_id(&self) -> impl Iterator<Item = GenericParamId> + '_ {
        self.iter_self().map(|(id, _)| id)
    }

    pub(crate) fn iter_parent_id(&self) -> impl Iterator<Item = GenericParamId> + '_ {
        self.iter_parent().map(|(id, _)| id)
    }

    pub(crate) fn iter_self_type_or_consts(
        &self,
    ) -> impl DoubleEndedIterator<Item = (LocalTypeOrConstParamId, &TypeOrConstParamData)> {
        self.params.iter_type_or_consts()
    }

    pub(crate) fn iter_self_type_or_consts_id(
        &self,
    ) -> impl DoubleEndedIterator<Item = GenericParamId> + '_ {
        self.params.iter_type_or_consts().map(from_toc_id(self)).map(TupleExt::head)
    }

    /// Iterate over the params followed by the parent params.
    pub(crate) fn iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = (GenericParamId, GenericParamDataRef<'_>)> + '_ {
        self.iter_self().chain(self.iter_parent())
    }

    pub(crate) fn iter_with_types_map(
        &self,
    ) -> impl Iterator<Item = ((GenericParamId, GenericParamDataRef<'_>), &TypesMap)> + '_ {
        self.iter_self().zip(std::iter::repeat(&self.params.types_map)).chain(
            self.iter_parent().zip(
                self.parent_generics()
                    .into_iter()
                    .flat_map(|it| std::iter::repeat(&it.params.types_map)),
            ),
        )
    }

    /// Iterate over the params without parent params.
    pub(crate) fn iter_self(
        &self,
    ) -> impl DoubleEndedIterator<Item = (GenericParamId, GenericParamDataRef<'_>)> + '_ {
        let mut toc = self.params.iter_type_or_consts().map(from_toc_id(self));
        let trait_self_param = self.has_trait_self_param.then(|| toc.next()).flatten();
        chain!(trait_self_param, self.params.iter_lt().map(from_lt_id(self)), toc)
    }

    /// Iterator over types and const params of parent.
    fn iter_parent(
        &self,
    ) -> impl DoubleEndedIterator<Item = (GenericParamId, GenericParamDataRef<'_>)> + '_ {
        self.parent_generics().into_iter().flat_map(|it| {
            let mut toc = it.params.iter_type_or_consts().map(from_toc_id(it));
            let trait_self_param = it.has_trait_self_param.then(|| toc.next()).flatten();
            chain!(trait_self_param, it.params.iter_lt().map(from_lt_id(it)), toc)
        })
    }

    /// Returns total number of generic parameters in scope, including those from parent.
    pub(crate) fn len(&self) -> usize {
        let parent = self.parent_generics().map_or(0, Generics::len);
        let child = self.params.len();
        parent + child
    }

    /// Returns numbers of generic parameters excluding those from parent.
    pub(crate) fn len_self(&self) -> usize {
        self.params.len()
    }

    /// (parent total, self param, type params, const params, impl trait list, lifetimes)
    pub(crate) fn provenance_split(&self) -> (usize, bool, usize, usize, usize, usize) {
        let mut self_param = false;
        let mut type_params = 0;
        let mut impl_trait_params = 0;
        let mut const_params = 0;
        self.params.iter_type_or_consts().for_each(|(_, data)| match data {
            TypeOrConstParamData::TypeParamData(p) => match p.provenance {
                TypeParamProvenance::TypeParamList => type_params += 1,
                TypeParamProvenance::TraitSelf => self_param |= true,
                TypeParamProvenance::ArgumentImplTrait => impl_trait_params += 1,
            },
            TypeOrConstParamData::ConstParamData(_) => const_params += 1,
        });

        let lifetime_params = self.params.iter_lt().count();

        let parent_len = self.parent_generics().map_or(0, Generics::len);
        (parent_len, self_param, type_params, const_params, impl_trait_params, lifetime_params)
    }

    pub(crate) fn type_or_const_param_idx(&self, param: TypeOrConstParamId) -> Option<usize> {
        self.find_type_or_const_param(param)
    }

    fn find_type_or_const_param(&self, param: TypeOrConstParamId) -> Option<usize> {
        if param.parent == self.def {
            let idx = param.local_id.into_raw().into_u32() as usize;
            debug_assert!(idx <= self.params.len_type_or_consts());
            if self.params.trait_self_param() == Some(param.local_id) {
                return Some(idx);
            }
            Some(self.params.len_lifetimes() + idx)
        } else {
            debug_assert_eq!(self.parent_generics().map(|it| it.def), Some(param.parent));
            self.parent_generics()
                .and_then(|g| g.find_type_or_const_param(param))
                // Remember that parent parameters come after parameters for self.
                .map(|idx| self.len_self() + idx)
        }
    }

    pub(crate) fn lifetime_idx(&self, lifetime: LifetimeParamId) -> Option<usize> {
        self.find_lifetime(lifetime)
    }

    fn find_lifetime(&self, lifetime: LifetimeParamId) -> Option<usize> {
        if lifetime.parent == self.def {
            let idx = lifetime.local_id.into_raw().into_u32() as usize;
            debug_assert!(idx <= self.params.len_lifetimes());
            Some(self.params.trait_self_param().is_some() as usize + idx)
        } else {
            debug_assert_eq!(self.parent_generics().map(|it| it.def), Some(lifetime.parent));
            self.parent_generics()
                .and_then(|g| g.find_lifetime(lifetime))
                .map(|idx| self.len_self() + idx)
        }
    }

    pub(crate) fn parent_generics(&self) -> Option<&Generics> {
        self.parent_generics.as_deref()
    }

    pub(crate) fn parent_or_self(&self) -> &Generics {
        self.parent_generics.as_deref().unwrap_or(self)
    }

    /// Returns a Substitution that replaces each parameter by a bound variable.
    pub(crate) fn bound_vars_subst(
        &self,
        db: &dyn HirDatabase,
        debruijn: DebruijnIndex,
    ) -> Substitution {
        Substitution::from_iter(
            Interner,
            self.iter_id().enumerate().map(|(idx, id)| match id {
                GenericParamId::ConstParamId(id) => BoundVar::new(debruijn, idx)
                    .to_const(Interner, db.const_param_ty(id))
                    .cast(Interner),
                GenericParamId::TypeParamId(_) => {
                    BoundVar::new(debruijn, idx).to_ty(Interner).cast(Interner)
                }
                GenericParamId::LifetimeParamId(_) => {
                    BoundVar::new(debruijn, idx).to_lifetime(Interner).cast(Interner)
                }
            }),
        )
    }

    /// Returns a Substitution that replaces each parameter by itself (i.e. `Ty::Param`).
    pub(crate) fn placeholder_subst(&self, db: &dyn HirDatabase) -> Substitution {
        Substitution::from_iter(
            Interner,
            self.iter_id().map(|id| match id {
                GenericParamId::TypeParamId(id) => {
                    to_placeholder_idx(db, id.into()).to_ty(Interner).cast(Interner)
                }
                GenericParamId::ConstParamId(id) => to_placeholder_idx(db, id.into())
                    .to_const(Interner, db.const_param_ty(id))
                    .cast(Interner),
                GenericParamId::LifetimeParamId(id) => {
                    lt_to_placeholder_idx(db, id).to_lifetime(Interner).cast(Interner)
                }
            }),
        )
    }
}

pub(crate) fn trait_self_param_idx(db: &dyn DefDatabase, def: GenericDefId) -> Option<usize> {
    match def {
        GenericDefId::TraitId(_) | GenericDefId::TraitAliasId(_) => {
            let params = db.generic_params(def);
            params.trait_self_param().map(|idx| idx.into_raw().into_u32() as usize)
        }
        GenericDefId::ImplId(_) => None,
        _ => {
            let parent_def = parent_generic_def(db, def)?;
            let parent_params = db.generic_params(parent_def);
            let parent_self_idx = parent_params.trait_self_param()?.into_raw().into_u32() as usize;
            let self_params = db.generic_params(def);
            Some(self_params.len() + parent_self_idx)
        }
    }
}

fn parent_generic_def(db: &dyn DefDatabase, def: GenericDefId) -> Option<GenericDefId> {
    let container = match def {
        GenericDefId::FunctionId(it) => it.lookup(db).container,
        GenericDefId::TypeAliasId(it) => it.lookup(db).container,
        GenericDefId::ConstId(it) => it.lookup(db).container,
        GenericDefId::AdtId(_)
        | GenericDefId::TraitId(_)
        | GenericDefId::ImplId(_)
        | GenericDefId::TraitAliasId(_) => return None,
    };

    match container {
        ItemContainerId::ImplId(it) => Some(it.into()),
        ItemContainerId::TraitId(it) => Some(it.into()),
        ItemContainerId::ModuleId(_) | ItemContainerId::ExternBlockId(_) => None,
    }
}

fn from_toc_id<'a>(
    it: &'a Generics,
) -> impl Fn(
    (LocalTypeOrConstParamId, &'a TypeOrConstParamData),
) -> (GenericParamId, GenericParamDataRef<'a>) {
    move |(local_id, p): (_, _)| {
        let id = TypeOrConstParamId { parent: it.def, local_id };
        match p {
            TypeOrConstParamData::TypeParamData(p) => (
                GenericParamId::TypeParamId(TypeParamId::from_unchecked(id)),
                GenericParamDataRef::TypeParamData(p),
            ),
            TypeOrConstParamData::ConstParamData(p) => (
                GenericParamId::ConstParamId(ConstParamId::from_unchecked(id)),
                GenericParamDataRef::ConstParamData(p),
            ),
        }
    }
}

fn from_lt_id<'a>(
    it: &'a Generics,
) -> impl Fn((LocalLifetimeParamId, &'a LifetimeParamData)) -> (GenericParamId, GenericParamDataRef<'a>)
{
    move |(local_id, p): (_, _)| {
        (
            GenericParamId::LifetimeParamId(LifetimeParamId { parent: it.def, local_id }),
            GenericParamDataRef::LifetimeParamData(p),
        )
    }
}
