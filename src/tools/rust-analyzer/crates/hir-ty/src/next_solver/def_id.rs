//! Definition of `SolverDefId`

use hir_def::{
    AdtId, AttrDefId, CallableDefId, ConstId, DefWithBodyId, EnumId, EnumVariantId, FunctionId,
    GeneralConstId, GenericDefId, HasModule, ImplId, ModuleId, StaticId, StructId, TraitId,
    TypeAliasId, UnionId, db::DefDatabase,
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

#[derive(PartialOrd, Ord, Clone, Copy, PartialEq, Eq, Hash, salsa::Supertype)]
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
    EnumVariantId(EnumVariantId),
    // FIXME(next-solver): Do we need the separation of `Ctor`? It duplicates some variants.
    Ctor(Ctor),
}

impl std::fmt::Debug for SolverDefId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let interner = DbInterner::conjure();
        let db = interner.db;
        match *self {
            SolverDefId::AdtId(AdtId::StructId(id)) => {
                f.debug_tuple("AdtId").field(&db.struct_signature(id).name.as_str()).finish()
            }
            SolverDefId::AdtId(AdtId::EnumId(id)) => {
                f.debug_tuple("AdtId").field(&db.enum_signature(id).name.as_str()).finish()
            }
            SolverDefId::AdtId(AdtId::UnionId(id)) => {
                f.debug_tuple("AdtId").field(&db.union_signature(id).name.as_str()).finish()
            }
            SolverDefId::ConstId(id) => f
                .debug_tuple("ConstId")
                .field(&db.const_signature(id).name.as_ref().map_or("_", |name| name.as_str()))
                .finish(),
            SolverDefId::FunctionId(id) => {
                f.debug_tuple("FunctionId").field(&db.function_signature(id).name.as_str()).finish()
            }
            SolverDefId::ImplId(id) => f.debug_tuple("ImplId").field(&id).finish(),
            SolverDefId::StaticId(id) => {
                f.debug_tuple("StaticId").field(&db.static_signature(id).name.as_str()).finish()
            }
            SolverDefId::TraitId(id) => {
                f.debug_tuple("TraitId").field(&db.trait_signature(id).name.as_str()).finish()
            }
            SolverDefId::TypeAliasId(id) => f
                .debug_tuple("TypeAliasId")
                .field(&db.type_alias_signature(id).name.as_str())
                .finish(),
            SolverDefId::InternedClosureId(id) => {
                f.debug_tuple("InternedClosureId").field(&id).finish()
            }
            SolverDefId::InternedCoroutineId(id) => {
                f.debug_tuple("InternedCoroutineId").field(&id).finish()
            }
            SolverDefId::InternedOpaqueTyId(id) => {
                f.debug_tuple("InternedOpaqueTyId").field(&id).finish()
            }
            SolverDefId::EnumVariantId(id) => {
                let parent_enum = id.loc(db).parent;
                f.debug_tuple("EnumVariantId")
                    .field(&format_args!(
                        "\"{}::{}\"",
                        db.enum_signature(parent_enum).name.as_str(),
                        parent_enum.enum_variants(db).variant_name_by_id(id).unwrap().as_str()
                    ))
                    .finish()
            }
            SolverDefId::Ctor(Ctor::Struct(id)) => {
                f.debug_tuple("Ctor").field(&db.struct_signature(id).name.as_str()).finish()
            }
            SolverDefId::Ctor(Ctor::Enum(id)) => {
                let parent_enum = id.loc(db).parent;
                f.debug_tuple("Ctor")
                    .field(&format_args!(
                        "\"{}::{}\"",
                        db.enum_signature(parent_enum).name.as_str(),
                        parent_enum.enum_variants(db).variant_name_by_id(id).unwrap().as_str()
                    ))
                    .finish()
            }
        }
    }
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
    EnumVariantId,
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

impl From<GeneralConstId> for SolverDefId {
    #[inline]
    fn from(value: GeneralConstId) -> Self {
        match value {
            GeneralConstId::ConstId(const_id) => SolverDefId::ConstId(const_id),
            GeneralConstId::StaticId(static_id) => SolverDefId::StaticId(static_id),
        }
    }
}

impl From<DefWithBodyId> for SolverDefId {
    #[inline]
    fn from(value: DefWithBodyId) -> Self {
        match value {
            DefWithBodyId::FunctionId(id) => id.into(),
            DefWithBodyId::StaticId(id) => id.into(),
            DefWithBodyId::ConstId(id) => id.into(),
            DefWithBodyId::VariantId(id) => id.into(),
        }
    }
}

impl TryFrom<SolverDefId> for AttrDefId {
    type Error = ();
    #[inline]
    fn try_from(value: SolverDefId) -> Result<Self, Self::Error> {
        match value {
            SolverDefId::AdtId(it) => Ok(it.into()),
            SolverDefId::ConstId(it) => Ok(it.into()),
            SolverDefId::FunctionId(it) => Ok(it.into()),
            SolverDefId::ImplId(it) => Ok(it.into()),
            SolverDefId::StaticId(it) => Ok(it.into()),
            SolverDefId::TraitId(it) => Ok(it.into()),
            SolverDefId::TypeAliasId(it) => Ok(it.into()),
            SolverDefId::EnumVariantId(it) => Ok(it.into()),
            SolverDefId::Ctor(Ctor::Struct(it)) => Ok(it.into()),
            SolverDefId::Ctor(Ctor::Enum(it)) => Ok(it.into()),
            SolverDefId::InternedClosureId(_)
            | SolverDefId::InternedCoroutineId(_)
            | SolverDefId::InternedOpaqueTyId(_) => Err(()),
        }
    }
}

impl TryFrom<SolverDefId> for DefWithBodyId {
    type Error = ();

    #[inline]
    fn try_from(value: SolverDefId) -> Result<Self, Self::Error> {
        let id = match value {
            SolverDefId::ConstId(id) => id.into(),
            SolverDefId::FunctionId(id) => id.into(),
            SolverDefId::StaticId(id) => id.into(),
            SolverDefId::EnumVariantId(id) | SolverDefId::Ctor(Ctor::Enum(id)) => id.into(),
            SolverDefId::InternedOpaqueTyId(_)
            | SolverDefId::TraitId(_)
            | SolverDefId::TypeAliasId(_)
            | SolverDefId::ImplId(_)
            | SolverDefId::InternedClosureId(_)
            | SolverDefId::InternedCoroutineId(_)
            | SolverDefId::Ctor(Ctor::Struct(_))
            | SolverDefId::AdtId(_) => return Err(()),
        };
        Ok(id)
    }
}

impl TryFrom<SolverDefId> for GenericDefId {
    type Error = ();

    fn try_from(value: SolverDefId) -> Result<Self, Self::Error> {
        Ok(match value {
            SolverDefId::AdtId(adt_id) => GenericDefId::AdtId(adt_id),
            SolverDefId::ConstId(const_id) => GenericDefId::ConstId(const_id),
            SolverDefId::FunctionId(function_id) => GenericDefId::FunctionId(function_id),
            SolverDefId::ImplId(impl_id) => GenericDefId::ImplId(impl_id),
            SolverDefId::StaticId(static_id) => GenericDefId::StaticId(static_id),
            SolverDefId::TraitId(trait_id) => GenericDefId::TraitId(trait_id),
            SolverDefId::TypeAliasId(type_alias_id) => GenericDefId::TypeAliasId(type_alias_id),
            SolverDefId::InternedClosureId(_)
            | SolverDefId::InternedCoroutineId(_)
            | SolverDefId::InternedOpaqueTyId(_)
            | SolverDefId::EnumVariantId(_)
            | SolverDefId::Ctor(_) => return Err(()),
        })
    }
}

impl SolverDefId {
    #[inline]
    #[track_caller]
    pub fn expect_opaque_ty(self) -> InternedOpaqueTyId {
        match self {
            SolverDefId::InternedOpaqueTyId(it) => it,
            _ => panic!("expected opaque type, found {self:?}"),
        }
    }

    #[inline]
    #[track_caller]
    pub fn expect_type_alias(self) -> TypeAliasId {
        match self {
            SolverDefId::TypeAliasId(it) => it,
            _ => panic!("expected type alias, found {self:?}"),
        }
    }
}

impl HasModule for SolverDefId {
    fn module(&self, db: &dyn DefDatabase) -> ModuleId {
        match *self {
            SolverDefId::AdtId(id) => id.module(db),
            SolverDefId::ConstId(id) => id.module(db),
            SolverDefId::FunctionId(id) => id.module(db),
            SolverDefId::ImplId(id) => id.module(db),
            SolverDefId::StaticId(id) => id.module(db),
            SolverDefId::TraitId(id) => id.module(db),
            SolverDefId::TypeAliasId(id) => id.module(db),
            SolverDefId::InternedClosureId(id) => id.loc(db).0.module(db),
            SolverDefId::InternedCoroutineId(id) => id.loc(db).0.module(db),
            SolverDefId::InternedOpaqueTyId(id) => match id.loc(db) {
                crate::ImplTraitId::ReturnTypeImplTrait(owner, _) => owner.module(db),
                crate::ImplTraitId::TypeAliasImplTrait(owner, _) => owner.module(db),
            },
            SolverDefId::Ctor(Ctor::Enum(id)) | SolverDefId::EnumVariantId(id) => id.module(db),
            SolverDefId::Ctor(Ctor::Struct(id)) => id.module(db),
        }
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
                std::fmt::Debug::fmt(&SolverDefId::from(self.0), f)
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
pub struct GeneralConstIdWrapper(pub GeneralConstId);

impl std::fmt::Debug for GeneralConstIdWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}
impl From<GeneralConstIdWrapper> for GeneralConstId {
    #[inline]
    fn from(value: GeneralConstIdWrapper) -> GeneralConstId {
        value.0
    }
}
impl From<GeneralConstId> for GeneralConstIdWrapper {
    #[inline]
    fn from(value: GeneralConstId) -> GeneralConstIdWrapper {
        Self(value)
    }
}
impl From<GeneralConstIdWrapper> for SolverDefId {
    #[inline]
    fn from(value: GeneralConstIdWrapper) -> SolverDefId {
        match value.0 {
            GeneralConstId::ConstId(id) => SolverDefId::ConstId(id),
            GeneralConstId::StaticId(id) => SolverDefId::StaticId(id),
        }
    }
}
impl TryFrom<SolverDefId> for GeneralConstIdWrapper {
    type Error = ();
    #[inline]
    fn try_from(value: SolverDefId) -> Result<Self, Self::Error> {
        match value {
            SolverDefId::ConstId(it) => Ok(Self(it.into())),
            SolverDefId::StaticId(it) => Ok(Self(it.into())),
            _ => Err(()),
        }
    }
}
impl<'db> inherent::DefId<DbInterner<'db>> for GeneralConstIdWrapper {
    fn as_local(self) -> Option<SolverDefId> {
        Some(self.into())
    }
    fn is_local(self) -> bool {
        true
    }
}

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
