//! Utilities for working with generics.
//!
//! The layout for generics as expected by chalk are as follows:
//! - Parent parameters
//! - Optional Self parameter
//! - Lifetime parameters
//! - Type or Const parameters
//!
//! where parent follows the same scheme.

use arrayvec::ArrayVec;
use base_db::SourceDatabase;
use hir_def::{
    ConstParamId, GenericDefId, GenericParamId, ItemContainerId, LifetimeParamId, Lookup,
    TypeOrConstParamId, TypeParamId,
    expr_store::ExpressionStore,
    hir::generics::{
        GenericParamDataRef, GenericParams, LifetimeParamData, TypeOrConstParamData,
        TypeParamProvenance, WherePredicate,
    },
};

pub(crate) fn generics(db: &dyn SourceDatabase, def: GenericDefId) -> Generics<'_> {
    let mut chain = ArrayVec::new();
    let mut parent_params_len = 0;
    if let Some(parent_def) = parent_generic_def(db, def) {
        let (parent_params, parent_store) = GenericParams::with_store(db, parent_def);
        chain.push(SingleGenerics {
            def: parent_def,
            params: parent_params,
            store: parent_store,
            preceding_params_len: 0,
        });
        parent_params_len = parent_params.len() as u32;
    }
    let (params, store) = GenericParams::with_store(db, def);
    chain.push(SingleGenerics { def, params, store, preceding_params_len: parent_params_len });
    Generics { chain }
}

#[derive(Debug)]
pub struct Generics<'db> {
    chain: ArrayVec<SingleGenerics<'db>, 2>,
}

#[derive(Debug)]
pub(crate) struct SingleGenerics<'db> {
    def: GenericDefId,
    preceding_params_len: u32,
    params: &'db GenericParams,
    store: &'db ExpressionStore,
}

impl<'db> SingleGenerics<'db> {
    pub(crate) fn def(&self) -> GenericDefId {
        self.def
    }

    pub(crate) fn store(&self) -> &'db ExpressionStore {
        self.store
    }

    pub(crate) fn where_predicates(&self) -> impl Iterator<Item = &WherePredicate> {
        self.params.where_predicates().iter()
    }

    pub(crate) fn has_no_params(&self) -> bool {
        self.params.is_empty()
    }

    pub(crate) fn len_lifetimes(&self) -> usize {
        self.params.len_lifetimes()
    }

    pub(crate) fn len(&self, consider_late_bound: bool) -> usize {
        if consider_late_bound {
            self.params.len()
        } else {
            self.params.len() - self.params.len_late_bound_lifetimes()
        }
    }

    fn iter_lifetimes(&self) -> impl Iterator<Item = (LifetimeParamId, &'db LifetimeParamData)> {
        let parent = self.def;
        self.params
            .iter_early_bound_lt()
            .map(move |(local_id, data)| (LifetimeParamId { parent, local_id }, data))
    }

    fn iter_late_bound_lifetimes(
        &self,
        consider_late_bound: bool,
    ) -> impl Iterator<Item = (LifetimeParamId, &'db LifetimeParamData)> {
        let parent = self.def;
        self.params.iter_late_bound_lt().filter_map(move |(local_id, data)| {
            consider_late_bound.then_some((LifetimeParamId { parent, local_id }, data))
        })
    }

    pub(crate) fn iter_type_or_consts(
        &self,
    ) -> impl Iterator<Item = (TypeOrConstParamId, &'db TypeOrConstParamData)> {
        let parent = self.def;
        self.params
            .iter_type_or_consts()
            .map(move |(local_id, data)| (TypeOrConstParamId { parent, local_id }, data))
    }

    fn iter_type_or_consts_as_generic(
        &self,
    ) -> impl Iterator<Item = (GenericParamId, GenericParamDataRef<'db>)> {
        self.iter_type_or_consts().map(|(id, data)| match data {
            TypeOrConstParamData::TypeParamData(data) => (
                GenericParamId::TypeParamId(TypeParamId::from_unchecked(id)),
                GenericParamDataRef::TypeParamData(data),
            ),
            TypeOrConstParamData::ConstParamData(data) => (
                GenericParamId::ConstParamId(ConstParamId::from_unchecked(id)),
                GenericParamDataRef::ConstParamData(data),
            ),
        })
    }

    fn trait_self_and_others(
        &self,
    ) -> (
        Option<(GenericParamId, GenericParamDataRef<'db>)>,
        impl Iterator<Item = (GenericParamId, GenericParamDataRef<'db>)>,
    ) {
        let mut iter = self.iter_type_or_consts_as_generic();
        let trait_self = if let GenericDefId::TraitId(_) = self.def { iter.next() } else { None };
        (trait_self, iter)
    }

    pub(crate) fn iter(
        &self,
        consider_late_bound: bool,
    ) -> impl Iterator<Item = (GenericParamId, GenericParamDataRef<'db>)> {
        let lifetime_map = |(id, data)| {
            (GenericParamId::LifetimeParamId(id), GenericParamDataRef::LifetimeParamData(data))
        };
        let lifetimes = self.iter_lifetimes().map(lifetime_map);
        let late_bound_lifetimes =
            self.iter_late_bound_lifetimes(consider_late_bound).map(lifetime_map);

        let (trait_self, type_and_consts) = self.trait_self_and_others();
        trait_self.into_iter().chain(lifetimes).chain(type_and_consts).chain(late_bound_lifetimes)
    }

    pub(crate) fn iter_with_idx(
        &self,
    ) -> impl Iterator<Item = (u32, GenericParamId, GenericParamDataRef<'db>)> {
        std::iter::zip(self.preceding_params_len.., self.iter(false))
            .map(|(index, (id, data))| (index, id, data))
    }

    pub(crate) fn iter_id(
        &self,
        consider_late_bound: bool,
    ) -> impl Iterator<Item = GenericParamId> {
        self.iter(consider_late_bound).map(|(id, _)| id)
    }

    pub(crate) fn iter_late_bound(
        &self,
    ) -> impl Iterator<Item = (GenericParamId, GenericParamDataRef<'db>)> {
        // we don't handle late bound types or const now, so it is ignored for now
        let parent = self.def;
        self.params.iter_late_bound_lt().map(move |(local_id, data)| {
            (
                GenericParamId::LifetimeParamId(LifetimeParamId { parent, local_id }),
                GenericParamDataRef::LifetimeParamData(data),
            )
        })
    }
}

impl<'db> Generics<'db> {
    pub(crate) fn iter_owners(&self) -> impl DoubleEndedIterator<Item = &SingleGenerics<'db>> {
        self.chain.iter()
    }

    pub(crate) fn owner(&self) -> &SingleGenerics<'db> {
        self.chain.last().expect("must have an owner params")
    }

    pub(crate) fn parent(&self) -> Option<&SingleGenerics<'db>> {
        match &*self.chain {
            [parent, _owner] => Some(parent),
            _ => None,
        }
    }

    pub(crate) fn has_no_params(&self) -> bool {
        self.iter_owners().all(|owner| owner.has_no_params())
    }

    pub(crate) fn def(&self) -> GenericDefId {
        self.owner().def
    }

    pub(crate) fn store(&self) -> &'db ExpressionStore {
        self.owner().store
    }

    pub(crate) fn iter_self(
        &self,
    ) -> impl Iterator<Item = (GenericParamId, GenericParamDataRef<'db>)> {
        self.owner().iter(false)
    }

    pub(crate) fn iter_self_with_idx(
        &self,
    ) -> impl Iterator<Item = (u32, GenericParamId, GenericParamDataRef<'db>)> {
        self.owner().iter_with_idx()
    }

    pub(crate) fn iter_self_late_bound(
        &self,
    ) -> impl Iterator<Item = (GenericParamId, GenericParamDataRef<'db>)> {
        self.owner().iter_late_bound()
    }

    pub(crate) fn iter_parent_id(&self) -> impl Iterator<Item = GenericParamId> {
        self.parent().into_iter().flat_map(move |parent| parent.iter_id(false))
    }

    pub(crate) fn iter_self_type_or_consts(
        &self,
    ) -> impl Iterator<Item = (TypeOrConstParamId, &'db TypeOrConstParamData)> {
        self.owner().iter_type_or_consts()
    }

    /// Iterate over the parent params followed by self params.
    pub(crate) fn iter(
        &self,
        consider_late_bound: bool,
    ) -> impl Iterator<Item = (GenericParamId, GenericParamDataRef<'db>)> {
        self.iter_owners().flat_map(move |owner| owner.iter(consider_late_bound))
    }

    pub(crate) fn iter_id(
        &self,
        consider_late_bound: bool,
    ) -> impl Iterator<Item = GenericParamId> {
        self.iter_owners().flat_map(move |owner| owner.iter_id(consider_late_bound))
    }

    /// Returns total number of generic parameters in scope, including those from parent.
    pub(crate) fn len(&self, consider_late_bound: bool) -> usize {
        match &*self.chain {
            [parent, owner] => parent.len(consider_late_bound) + owner.len(consider_late_bound),
            [owner] => owner.len(consider_late_bound),
            _ => unreachable!(),
        }
    }

    #[inline]
    pub(crate) fn len_parent(&self) -> usize {
        // add `consider_late_bound` arg if needed in future, currently it's not needed.
        self.parent().map_or(0, |p| p.len(true))
    }

    pub(crate) fn len_lifetimes_self(&self) -> usize {
        self.owner().len_lifetimes()
    }

    pub(crate) fn provenance_split(&self) -> ProvenanceSplit {
        let parent_total = self.len_parent();

        let owner = self.owner();
        let lifetimes = owner.params.len_lifetimes();

        let mut has_self_param = false;
        let mut non_impl_trait_type_params = 0;
        let mut impl_trait_type_params = 0;
        let mut const_params = 0;
        owner.params.iter_type_or_consts().for_each(|(_, data)| match data {
            TypeOrConstParamData::TypeParamData(p) => match p.provenance {
                TypeParamProvenance::TypeParamList => non_impl_trait_type_params += 1,
                TypeParamProvenance::TraitSelf => has_self_param |= true,
                TypeParamProvenance::ArgumentImplTrait => impl_trait_type_params += 1,
            },
            TypeOrConstParamData::ConstParamData(_) => const_params += 1,
        });

        ProvenanceSplit {
            parent_total,
            has_self_param,
            non_impl_trait_type_params,
            const_params,
            impl_trait_type_params,
            lifetimes,
        }
    }

    fn find_owner(&self, def: GenericDefId) -> &SingleGenerics<'db> {
        match &*self.chain {
            [parent, owner] => {
                if parent.def == def {
                    parent
                } else {
                    debug_assert_eq!(def, owner.def);
                    owner
                }
            }
            [owner] => {
                debug_assert_eq!(def, owner.def);
                owner
            }
            _ => unreachable!(),
        }
    }

    pub(crate) fn type_or_const_param_idx(&self, param: TypeOrConstParamId) -> u32 {
        let owner = self.find_owner(param.parent);
        let has_trait_self = matches!(owner.def, GenericDefId::TraitId(_));
        if has_trait_self && param.local_id == GenericParams::SELF_PARAM_ID_IN_SELF {
            owner.preceding_params_len
        } else {
            owner.preceding_params_len
                + owner.len_lifetimes() as u32
                + param.local_id.into_raw().into_u32()
        }
    }

    // Rename this?
    pub(crate) fn lifetime_param_idx(
        &self,
        param: LifetimeParamId,
        is_lowering_impl_trait_bounds: bool,
    ) -> (u32, bool) {
        let owner = self.find_owner(param.parent);
        if is_lowering_impl_trait_bounds {
            let idx = self.opaque_lifetime_idx(param);
            return (owner.preceding_params_len + (idx as u32), false);
        }

        let has_trait_self = matches!(owner.def, GenericDefId::TraitId(_));
        match owner.params.lifetime_param_idx(&param.local_id) {
            Some((idx, is_late_bound)) => {
                let idx = if is_late_bound {
                    idx as u32
                } else {
                    owner.preceding_params_len + u32::from(has_trait_self) + (idx as u32)
                };
                (idx, is_late_bound)
            }
            _ => unreachable!(),
        }
    }

    #[deprecated = "don't use this; it's easy to expose an erroneous `Generics` with this"]
    pub(crate) fn empty(def: GenericDefId) -> Self {
        let mut chain = ArrayVec::new();
        chain.push(SingleGenerics {
            def,
            preceding_params_len: 0,
            params: GenericParams::empty(),
            store: ExpressionStore::empty(),
        });
        Generics { chain }
    }

    fn opaque_lifetime_idx(&self, param: LifetimeParamId) -> usize {
        self.find_owner(param.parent)
            .iter_id(true)
            .position(|id| {
                let GenericParamId::LifetimeParamId(id) = id else {
                    return false;
                };
                param == id
            })
            .unwrap()
    }
}

pub(crate) struct ProvenanceSplit {
    pub(crate) parent_total: usize,
    // The rest are about self.
    pub(crate) has_self_param: bool,
    pub(crate) non_impl_trait_type_params: usize,
    pub(crate) const_params: usize,
    pub(crate) impl_trait_type_params: usize,
    pub(crate) lifetimes: usize,
}

fn parent_generic_def(db: &dyn SourceDatabase, def: GenericDefId) -> Option<GenericDefId> {
    let container = match def {
        GenericDefId::FunctionId(it) => it.lookup(db).container,
        GenericDefId::TypeAliasId(it) => it.lookup(db).container,
        GenericDefId::ConstId(it) => it.lookup(db).container,
        GenericDefId::StaticId(_)
        | GenericDefId::AdtId(_)
        | GenericDefId::TraitId(_)
        | GenericDefId::ImplId(_) => return None,
    };

    match container {
        ItemContainerId::ImplId(it) => Some(it.into()),
        ItemContainerId::TraitId(it) => Some(it.into()),
        ItemContainerId::ModuleId(_) | ItemContainerId::ExternBlockId(_) => None,
    }
}
