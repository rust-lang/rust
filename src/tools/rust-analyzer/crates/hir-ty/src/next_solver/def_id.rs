//! Definition of `SolverDefId`

use hir_def::{
    AdtId, ConstId, EnumId, EnumVariantId, FunctionId, GenericDefId, ImplId, StaticId, StructId,
    TraitId, TypeAliasId, UnionId,
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
    ForeignId(TypeAliasId),
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
    InternedOpaqueTyId
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
            SolverDefId::ForeignId(_) => return Err(value),
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
