//! Definition of `SolverDefId`

use hir_def::{
    AdtId, CallableDefId, ConstId, EnumId, EnumVariantId, FunctionId, GenericDefId, ImplId,
    StaticId, StructId, TraitId, TypeAliasId, UnionId,
};
use rustc_type_ir::inherent;
use stdx::impl_from;

use crate::db::{InternedClosureId, InternedCoroutineId, InternedOpaqueTyId};

use super::DbInterner;

#[derive(Debug, PartialOrd, Ord, Clone, Copy, PartialEq, Eq, Hash, salsa::Supertype)]
pub enum Ctor {
    Struct(StructId),
    Enum(EnumVariantId),
}

#[derive(Debug, PartialOrd, Ord, Clone, Copy, PartialEq, Eq, Hash, salsa::Supertype)]
pub enum SolverDefId {
    AdtId(AdtId),
    ConstId(ConstId),
    FunctionId(FunctionId),
    ImplId(ImplId),
    StaticId(StaticId),
    TraitId(TraitId),
    TypeAliasId(TypeAliasId),
    InternedClosureId(InternedClosureId),
    InternedCoroutineId(InternedCoroutineId),
    InternedOpaqueTyId(InternedOpaqueTyId),
    Ctor(Ctor),
}

impl_from!(
    AdtId(StructId, EnumId, UnionId),
    ConstId,
    FunctionId,
    ImplId,
    StaticId,
    TraitId,
    TypeAliasId,
    InternedClosureId,
    InternedCoroutineId,
    InternedOpaqueTyId,
    Ctor
    for SolverDefId
);

impl From<GenericDefId> for SolverDefId {
    fn from(value: GenericDefId) -> Self {
        match value {
            GenericDefId::AdtId(adt_id) => SolverDefId::AdtId(adt_id),
            GenericDefId::ConstId(const_id) => SolverDefId::ConstId(const_id),
            GenericDefId::FunctionId(function_id) => SolverDefId::FunctionId(function_id),
            GenericDefId::ImplId(impl_id) => SolverDefId::ImplId(impl_id),
            GenericDefId::StaticId(static_id) => SolverDefId::StaticId(static_id),
            GenericDefId::TraitId(trait_id) => SolverDefId::TraitId(trait_id),
            GenericDefId::TypeAliasId(type_alias_id) => SolverDefId::TypeAliasId(type_alias_id),
        }
    }
}

impl TryFrom<SolverDefId> for GenericDefId {
    type Error = SolverDefId;

    fn try_from(value: SolverDefId) -> Result<Self, Self::Error> {
        Ok(match value {
            SolverDefId::AdtId(adt_id) => GenericDefId::AdtId(adt_id),
            SolverDefId::ConstId(const_id) => GenericDefId::ConstId(const_id),
            SolverDefId::FunctionId(function_id) => GenericDefId::FunctionId(function_id),
            SolverDefId::ImplId(impl_id) => GenericDefId::ImplId(impl_id),
            SolverDefId::StaticId(static_id) => GenericDefId::StaticId(static_id),
            SolverDefId::TraitId(trait_id) => GenericDefId::TraitId(trait_id),
            SolverDefId::TypeAliasId(type_alias_id) => GenericDefId::TypeAliasId(type_alias_id),
            SolverDefId::InternedClosureId(_) => return Err(value),
            SolverDefId::InternedCoroutineId(_) => return Err(value),
            SolverDefId::InternedOpaqueTyId(_) => return Err(value),
            SolverDefId::Ctor(_) => return Err(value),
        })
    }
}

impl<'db> inherent::DefId<DbInterner<'db>> for SolverDefId {
    fn as_local(self) -> Option<SolverDefId> {
        Some(self)
    }
    fn is_local(self) -> bool {
        true
    }
}

macro_rules! declare_id_wrapper {
    ($name:ident, $wraps:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(pub $wraps);

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Debug::fmt(&self.0, f)
            }
        }

        impl From<$name> for $wraps {
            #[inline]
            fn from(value: $name) -> $wraps {
                value.0
            }
        }

        impl From<$wraps> for $name {
            #[inline]
            fn from(value: $wraps) -> $name {
                Self(value)
            }
        }

        impl From<$name> for SolverDefId {
            #[inline]
            fn from(value: $name) -> SolverDefId {
                value.0.into()
            }
        }

        impl TryFrom<SolverDefId> for $name {
            type Error = ();

            #[inline]
            fn try_from(value: SolverDefId) -> Result<Self, Self::Error> {
                match value {
                    SolverDefId::$wraps(it) => Ok(Self(it)),
                    _ => Err(()),
                }
            }
        }

        impl<'db> inherent::DefId<DbInterner<'db>> for $name {
            fn as_local(self) -> Option<SolverDefId> {
                Some(self.into())
            }
            fn is_local(self) -> bool {
                true
            }
        }
    };
}

declare_id_wrapper!(TraitIdWrapper, TraitId);
declare_id_wrapper!(TypeAliasIdWrapper, TypeAliasId);
declare_id_wrapper!(ClosureIdWrapper, InternedClosureId);
declare_id_wrapper!(CoroutineIdWrapper, InternedCoroutineId);
declare_id_wrapper!(AdtIdWrapper, AdtId);
declare_id_wrapper!(ImplIdWrapper, ImplId);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallableIdWrapper(pub CallableDefId);

impl std::fmt::Debug for CallableIdWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}
impl From<CallableIdWrapper> for CallableDefId {
    #[inline]
    fn from(value: CallableIdWrapper) -> CallableDefId {
        value.0
    }
}
impl From<CallableDefId> for CallableIdWrapper {
    #[inline]
    fn from(value: CallableDefId) -> CallableIdWrapper {
        Self(value)
    }
}
impl From<CallableIdWrapper> for SolverDefId {
    #[inline]
    fn from(value: CallableIdWrapper) -> SolverDefId {
        match value.0 {
            CallableDefId::FunctionId(it) => it.into(),
            CallableDefId::StructId(it) => Ctor::Struct(it).into(),
            CallableDefId::EnumVariantId(it) => Ctor::Enum(it).into(),
        }
    }
}
impl TryFrom<SolverDefId> for CallableIdWrapper {
    type Error = ();
    #[inline]
    fn try_from(value: SolverDefId) -> Result<Self, Self::Error> {
        match value {
            SolverDefId::FunctionId(it) => Ok(Self(it.into())),
            SolverDefId::Ctor(Ctor::Struct(it)) => Ok(Self(it.into())),
            SolverDefId::Ctor(Ctor::Enum(it)) => Ok(Self(it.into())),
            _ => Err(()),
        }
    }
}
impl<'db> inherent::DefId<DbInterner<'db>> for CallableIdWrapper {
    fn as_local(self) -> Option<SolverDefId> {
        Some(self.into())
    }
    fn is_local(self) -> bool {
        true
    }
}
