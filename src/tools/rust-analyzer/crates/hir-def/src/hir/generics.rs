//! Pre-type IR item generics
use std::{ops, sync::LazyLock};

use hir_expand::name::Name;
use la_arena::{Arena, Idx, RawIdx};
use stdx::impl_from;
use thin_vec::ThinVec;
use triomphe::Arc;

use crate::{
    AdtId, ConstParamId, GenericDefId, LifetimeParamId, TypeOrConstParamId, TypeParamId,
    db::DefDatabase,
    expr_store::{ExpressionStore, ExpressionStoreSourceMap},
    type_ref::{ConstRef, LifetimeRefId, TypeBound, TypeRefId},
};

pub type LocalTypeOrConstParamId = Idx<TypeOrConstParamData>;
pub type LocalLifetimeParamId = Idx<LifetimeParamData>;

/// Data about a generic type parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct TypeParamData {
    /// [`None`] only if the type ref is an [`TypeRef::ImplTrait`]. FIXME: Might be better to just
    /// make it always be a value, giving impl trait a special name.
    pub name: Option<Name>,
    pub default: Option<TypeRefId>,
    pub provenance: TypeParamProvenance,
}

/// Data about a generic lifetime parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct LifetimeParamData {
    pub name: Name,
}

/// Data about a generic const parameter (to a function, struct, impl, ...).
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct ConstParamData {
    pub name: Name,
    pub ty: TypeRefId,
    pub default: Option<ConstRef>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum TypeParamProvenance {
    TypeParamList,
    TraitSelf,
    ArgumentImplTrait,
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum TypeOrConstParamData {
    TypeParamData(TypeParamData),
    ConstParamData(ConstParamData),
}

impl TypeOrConstParamData {
    pub fn name(&self) -> Option<&Name> {
        match self {
            TypeOrConstParamData::TypeParamData(it) => it.name.as_ref(),
            TypeOrConstParamData::ConstParamData(it) => Some(&it.name),
        }
    }

    pub fn has_default(&self) -> bool {
        match self {
            TypeOrConstParamData::TypeParamData(it) => it.default.is_some(),
            TypeOrConstParamData::ConstParamData(it) => it.default.is_some(),
        }
    }

    pub fn type_param(&self) -> Option<&TypeParamData> {
        match self {
            TypeOrConstParamData::TypeParamData(it) => Some(it),
            TypeOrConstParamData::ConstParamData(_) => None,
        }
    }

    pub fn const_param(&self) -> Option<&ConstParamData> {
        match self {
            TypeOrConstParamData::TypeParamData(_) => None,
            TypeOrConstParamData::ConstParamData(it) => Some(it),
        }
    }

    pub fn is_trait_self(&self) -> bool {
        match self {
            TypeOrConstParamData::TypeParamData(it) => {
                it.provenance == TypeParamProvenance::TraitSelf
            }
            TypeOrConstParamData::ConstParamData(_) => false,
        }
    }
}

impl_from!(TypeParamData, ConstParamData for TypeOrConstParamData);

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum GenericParamData {
    TypeParamData(TypeParamData),
    ConstParamData(ConstParamData),
    LifetimeParamData(LifetimeParamData),
}

impl GenericParamData {
    pub fn name(&self) -> Option<&Name> {
        match self {
            GenericParamData::TypeParamData(it) => it.name.as_ref(),
            GenericParamData::ConstParamData(it) => Some(&it.name),
            GenericParamData::LifetimeParamData(it) => Some(&it.name),
        }
    }

    pub fn type_param(&self) -> Option<&TypeParamData> {
        match self {
            GenericParamData::TypeParamData(it) => Some(it),
            _ => None,
        }
    }

    pub fn const_param(&self) -> Option<&ConstParamData> {
        match self {
            GenericParamData::ConstParamData(it) => Some(it),
            _ => None,
        }
    }

    pub fn lifetime_param(&self) -> Option<&LifetimeParamData> {
        match self {
            GenericParamData::LifetimeParamData(it) => Some(it),
            _ => None,
        }
    }
}

impl_from!(TypeParamData, ConstParamData, LifetimeParamData for GenericParamData);

#[derive(Debug, Clone, Copy)]
pub enum GenericParamDataRef<'a> {
    TypeParamData(&'a TypeParamData),
    ConstParamData(&'a ConstParamData),
    LifetimeParamData(&'a LifetimeParamData),
}

/// Data about the generic parameters of a function, struct, impl, etc.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct GenericParams {
    pub(crate) type_or_consts: Arena<TypeOrConstParamData>,
    pub(crate) lifetimes: Arena<LifetimeParamData>,
    pub(crate) where_predicates: Box<[WherePredicate]>,
}

impl ops::Index<LocalTypeOrConstParamId> for GenericParams {
    type Output = TypeOrConstParamData;
    fn index(&self, index: LocalTypeOrConstParamId) -> &TypeOrConstParamData {
        &self.type_or_consts[index]
    }
}

impl ops::Index<LocalLifetimeParamId> for GenericParams {
    type Output = LifetimeParamData;
    fn index(&self, index: LocalLifetimeParamId) -> &LifetimeParamData {
        &self.lifetimes[index]
    }
}

/// A single predicate from a where clause, i.e. `where Type: Trait`. Combined
/// where clauses like `where T: Foo + Bar` are turned into multiple of these.
/// It might still result in multiple actual predicates though, because of
/// associated type bindings like `Iterator<Item = u32>`.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum WherePredicate {
    TypeBound { target: TypeRefId, bound: TypeBound },
    Lifetime { target: LifetimeRefId, bound: LifetimeRefId },
    ForLifetime { lifetimes: ThinVec<Name>, target: TypeRefId, bound: TypeBound },
}

static EMPTY: LazyLock<Arc<GenericParams>> = LazyLock::new(|| {
    Arc::new(GenericParams {
        type_or_consts: Arena::default(),
        lifetimes: Arena::default(),
        where_predicates: Box::default(),
    })
});

impl GenericParams {
    /// The index of the self param in the generic of the non-parent definition.
    pub(crate) const SELF_PARAM_ID_IN_SELF: la_arena::Idx<TypeOrConstParamData> =
        LocalTypeOrConstParamId::from_raw(RawIdx::from_u32(0));

    pub fn new(db: &dyn DefDatabase, def: GenericDefId) -> Arc<GenericParams> {
        match def {
            GenericDefId::AdtId(AdtId::EnumId(it)) => db.enum_signature(it).generic_params.clone(),
            GenericDefId::AdtId(AdtId::StructId(it)) => {
                db.struct_signature(it).generic_params.clone()
            }
            GenericDefId::AdtId(AdtId::UnionId(it)) => {
                db.union_signature(it).generic_params.clone()
            }
            GenericDefId::ConstId(_) => EMPTY.clone(),
            GenericDefId::FunctionId(function_id) => {
                db.function_signature(function_id).generic_params.clone()
            }
            GenericDefId::ImplId(impl_id) => db.impl_signature(impl_id).generic_params.clone(),
            GenericDefId::StaticId(_) => EMPTY.clone(),
            GenericDefId::TraitAliasId(trait_alias_id) => {
                db.trait_alias_signature(trait_alias_id).generic_params.clone()
            }
            GenericDefId::TraitId(trait_id) => db.trait_signature(trait_id).generic_params.clone(),
            GenericDefId::TypeAliasId(type_alias_id) => {
                db.type_alias_signature(type_alias_id).generic_params.clone()
            }
        }
    }

    pub fn generic_params_and_store(
        db: &dyn DefDatabase,
        def: GenericDefId,
    ) -> (Arc<GenericParams>, Arc<ExpressionStore>) {
        match def {
            GenericDefId::AdtId(AdtId::EnumId(id)) => {
                let sig = db.enum_signature(id);
                (sig.generic_params.clone(), sig.store.clone())
            }
            GenericDefId::AdtId(AdtId::StructId(id)) => {
                let sig = db.struct_signature(id);
                (sig.generic_params.clone(), sig.store.clone())
            }
            GenericDefId::AdtId(AdtId::UnionId(id)) => {
                let sig = db.union_signature(id);
                (sig.generic_params.clone(), sig.store.clone())
            }
            GenericDefId::ConstId(id) => {
                let sig = db.const_signature(id);
                (EMPTY.clone(), sig.store.clone())
            }
            GenericDefId::FunctionId(id) => {
                let sig = db.function_signature(id);
                (sig.generic_params.clone(), sig.store.clone())
            }
            GenericDefId::ImplId(id) => {
                let sig = db.impl_signature(id);
                (sig.generic_params.clone(), sig.store.clone())
            }
            GenericDefId::StaticId(id) => {
                let sig = db.static_signature(id);
                (EMPTY.clone(), sig.store.clone())
            }
            GenericDefId::TraitAliasId(id) => {
                let sig = db.trait_alias_signature(id);
                (sig.generic_params.clone(), sig.store.clone())
            }
            GenericDefId::TraitId(id) => {
                let sig = db.trait_signature(id);
                (sig.generic_params.clone(), sig.store.clone())
            }
            GenericDefId::TypeAliasId(id) => {
                let sig = db.type_alias_signature(id);
                (sig.generic_params.clone(), sig.store.clone())
            }
        }
    }

    pub fn generic_params_and_store_and_source_map(
        db: &dyn DefDatabase,
        def: GenericDefId,
    ) -> (Arc<GenericParams>, Arc<ExpressionStore>, Arc<ExpressionStoreSourceMap>) {
        match def {
            GenericDefId::AdtId(AdtId::EnumId(id)) => {
                let (sig, sm) = db.enum_signature_with_source_map(id);
                (sig.generic_params.clone(), sig.store.clone(), sm)
            }
            GenericDefId::AdtId(AdtId::StructId(id)) => {
                let (sig, sm) = db.struct_signature_with_source_map(id);
                (sig.generic_params.clone(), sig.store.clone(), sm)
            }
            GenericDefId::AdtId(AdtId::UnionId(id)) => {
                let (sig, sm) = db.union_signature_with_source_map(id);
                (sig.generic_params.clone(), sig.store.clone(), sm)
            }
            GenericDefId::ConstId(id) => {
                let (sig, sm) = db.const_signature_with_source_map(id);
                (EMPTY.clone(), sig.store.clone(), sm)
            }
            GenericDefId::FunctionId(id) => {
                let (sig, sm) = db.function_signature_with_source_map(id);
                (sig.generic_params.clone(), sig.store.clone(), sm)
            }
            GenericDefId::ImplId(id) => {
                let (sig, sm) = db.impl_signature_with_source_map(id);
                (sig.generic_params.clone(), sig.store.clone(), sm)
            }
            GenericDefId::StaticId(id) => {
                let (sig, sm) = db.static_signature_with_source_map(id);
                (EMPTY.clone(), sig.store.clone(), sm)
            }
            GenericDefId::TraitAliasId(id) => {
                let (sig, sm) = db.trait_alias_signature_with_source_map(id);
                (sig.generic_params.clone(), sig.store.clone(), sm)
            }
            GenericDefId::TraitId(id) => {
                let (sig, sm) = db.trait_signature_with_source_map(id);
                (sig.generic_params.clone(), sig.store.clone(), sm)
            }
            GenericDefId::TypeAliasId(id) => {
                let (sig, sm) = db.type_alias_signature_with_source_map(id);
                (sig.generic_params.clone(), sig.store.clone(), sm)
            }
        }
    }

    /// Number of Generic parameters (type_or_consts + lifetimes)
    #[inline]
    pub fn len(&self) -> usize {
        self.type_or_consts.len() + self.lifetimes.len()
    }

    #[inline]
    pub fn len_lifetimes(&self) -> usize {
        self.lifetimes.len()
    }

    #[inline]
    pub fn len_type_or_consts(&self) -> usize {
        self.type_or_consts.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn has_no_predicates(&self) -> bool {
        self.where_predicates.is_empty()
    }

    #[inline]
    pub fn where_predicates(&self) -> &[WherePredicate] {
        &self.where_predicates
    }

    /// Iterator of type_or_consts field
    #[inline]
    pub fn iter_type_or_consts(
        &self,
    ) -> impl DoubleEndedIterator<Item = (LocalTypeOrConstParamId, &TypeOrConstParamData)> {
        self.type_or_consts.iter()
    }

    /// Iterator of lifetimes field
    #[inline]
    pub fn iter_lt(
        &self,
    ) -> impl DoubleEndedIterator<Item = (LocalLifetimeParamId, &LifetimeParamData)> {
        self.lifetimes.iter()
    }

    pub fn find_type_by_name(&self, name: &Name, parent: GenericDefId) -> Option<TypeParamId> {
        self.type_or_consts.iter().find_map(|(id, p)| {
            if p.name().as_ref() == Some(&name) && p.type_param().is_some() {
                Some(TypeParamId::from_unchecked(TypeOrConstParamId { local_id: id, parent }))
            } else {
                None
            }
        })
    }

    pub fn find_const_by_name(&self, name: &Name, parent: GenericDefId) -> Option<ConstParamId> {
        self.type_or_consts.iter().find_map(|(id, p)| {
            if p.name().as_ref() == Some(&name) && p.const_param().is_some() {
                Some(ConstParamId::from_unchecked(TypeOrConstParamId { local_id: id, parent }))
            } else {
                None
            }
        })
    }

    #[inline]
    pub fn trait_self_param(&self) -> Option<LocalTypeOrConstParamId> {
        if self.type_or_consts.is_empty() {
            return None;
        }
        matches!(
            self.type_or_consts[Self::SELF_PARAM_ID_IN_SELF],
            TypeOrConstParamData::TypeParamData(TypeParamData {
                provenance: TypeParamProvenance::TraitSelf,
                ..
            })
        )
        .then(|| Self::SELF_PARAM_ID_IN_SELF)
    }

    pub fn find_lifetime_by_name(
        &self,
        name: &Name,
        parent: GenericDefId,
    ) -> Option<LifetimeParamId> {
        self.lifetimes.iter().find_map(|(id, p)| {
            if &p.name == name { Some(LifetimeParamId { local_id: id, parent }) } else { None }
        })
    }
}
