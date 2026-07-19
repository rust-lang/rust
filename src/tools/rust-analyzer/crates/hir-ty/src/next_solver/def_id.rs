//! Definition of `SolverDefId`

use hir_def::{
    AdtId, AttrDefId, BuiltinDeriveImplId, CallableDefId, ConstId, DefWithBodyId, EnumId,
    EnumVariantId, ExpressionStoreOwnerId, FunctionId, GenericDefId, ImplId, StaticId, StructId,
    TraitId, TypeAliasId, UnionId, VariantId,
    signatures::{
        ConstSignature, EnumSignature, FunctionSignature, StaticSignature, StructSignature,
        TraitSignature, TypeAliasSignature, UnionSignature,
    },
};
use rustc_type_ir::inherent;
use salsa::Update;
use stdx::impl_from;

use crate::{
    InferBodyId,
    db::{
        AnonConstId, GeneralConstId, InternedClosureId, InternedCoroutineClosureId,
        InternedCoroutineId, InternedOpaqueTyId,
    },
};

use super::DbInterner;

#[derive(Debug, PartialOrd, Ord, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ctor {
    Struct(StructId),
    Enum(EnumVariantId),
}

#[derive(PartialOrd, Ord, Clone, Copy, PartialEq, Eq, Hash, Update)]
pub enum SolverDefId<'db> {
    AdtId(AdtId),
    ConstId(ConstId),
    FunctionId(FunctionId),
    ImplId(ImplId),
    BuiltinDeriveImplId(BuiltinDeriveImplId),
    StaticId(StaticId),
    AnonConstId(AnonConstId<'db>),
    TraitId(TraitId),
    TypeAliasId(TypeAliasId),
    InternedClosureId(InternedClosureId<'db>),
    InternedCoroutineId(InternedCoroutineId<'db>),
    InternedCoroutineClosureId(InternedCoroutineClosureId<'db>),
    InternedOpaqueTyId(InternedOpaqueTyId),
    EnumVariantId(EnumVariantId),
    Ctor(Ctor),
}

impl std::fmt::Debug for SolverDefId<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let interner = DbInterner::conjure();
        let db = interner.db;
        match *self {
            SolverDefId::AdtId(AdtId::StructId(id)) => {
                f.debug_tuple("AdtId").field(&StructSignature::of(db, id).name.as_str()).finish()
            }
            SolverDefId::AdtId(AdtId::EnumId(id)) => {
                f.debug_tuple("AdtId").field(&EnumSignature::of(db, id).name.as_str()).finish()
            }
            SolverDefId::AdtId(AdtId::UnionId(id)) => {
                f.debug_tuple("AdtId").field(&UnionSignature::of(db, id).name.as_str()).finish()
            }
            SolverDefId::ConstId(id) => f
                .debug_tuple("ConstId")
                .field(&ConstSignature::of(db, id).name.as_ref().map_or("_", |name| name.as_str()))
                .finish(),
            SolverDefId::FunctionId(id) => f
                .debug_tuple("FunctionId")
                .field(&FunctionSignature::of(db, id).name.as_str())
                .finish(),
            SolverDefId::ImplId(id) => f.debug_tuple("ImplId").field(&id).finish(),
            SolverDefId::BuiltinDeriveImplId(id) => f.debug_tuple("ImplId").field(&id).finish(),
            SolverDefId::StaticId(id) => {
                f.debug_tuple("StaticId").field(&StaticSignature::of(db, id).name.as_str()).finish()
            }
            SolverDefId::TraitId(id) => {
                f.debug_tuple("TraitId").field(&TraitSignature::of(db, id).name.as_str()).finish()
            }
            SolverDefId::TypeAliasId(id) => f
                .debug_tuple("TypeAliasId")
                .field(&TypeAliasSignature::of(db, id).name.as_str())
                .finish(),
            SolverDefId::InternedClosureId(id) => {
                f.debug_tuple("InternedClosureId").field(&id).finish()
            }
            SolverDefId::InternedCoroutineId(id) => {
                f.debug_tuple("InternedCoroutineId").field(&id).finish()
            }
            SolverDefId::InternedCoroutineClosureId(id) => {
                f.debug_tuple("InternedCoroutineClosureId").field(&id).finish()
            }
            SolverDefId::InternedOpaqueTyId(id) => {
                f.debug_tuple("InternedOpaqueTyId").field(&id).finish()
            }
            SolverDefId::EnumVariantId(id) => {
                let parent_enum = id.loc(db).parent;
                f.debug_tuple("EnumVariantId")
                    .field(&format_args!(
                        "\"{}::{}\"",
                        EnumSignature::of(db, parent_enum).name.as_str(),
                        parent_enum.enum_variants(db).variant_name_by_id(id).unwrap().as_str()
                    ))
                    .finish()
            }
            SolverDefId::AnonConstId(id) => f.debug_tuple("AnonConstId").field(&id).finish(),
            SolverDefId::Ctor(Ctor::Struct(id)) => {
                f.debug_tuple("Ctor").field(&StructSignature::of(db, id).name.as_str()).finish()
            }
            SolverDefId::Ctor(Ctor::Enum(id)) => {
                let parent_enum = id.loc(db).parent;
                f.debug_tuple("Ctor")
                    .field(&format_args!(
                        "\"{}::{}\"",
                        EnumSignature::of(db, parent_enum).name.as_str(),
                        parent_enum.enum_variants(db).variant_name_by_id(id).unwrap().as_str()
                    ))
                    .finish()
            }
        }
    }
}

impl_from!(
    impl<'db>
    AdtId(StructId, EnumId, UnionId),
    ConstId,
    FunctionId,
    ImplId,
    BuiltinDeriveImplId,
    StaticId,
    AnonConstId<'db>,
    TraitId,
    TypeAliasId,
    InternedClosureId<'db>,
    InternedCoroutineId<'db>,
    InternedCoroutineClosureId<'db>,
    InternedOpaqueTyId,
    EnumVariantId,
    Ctor
    for SolverDefId<'db>
);

impl_from!(
    impl<'db>
    GenericDefId { AdtId, ConstId, FunctionId, ImplId, StaticId, TraitId, TypeAliasId }
    for SolverDefId<'db>
);

impl<'db> From<GeneralConstId<'db>> for SolverDefId<'db> {
    #[inline]
    fn from(value: GeneralConstId<'db>) -> Self {
        match value {
            GeneralConstId::ConstId(const_id) => SolverDefId::ConstId(const_id),
            GeneralConstId::StaticId(static_id) => SolverDefId::StaticId(static_id),
            GeneralConstId::AnonConstId(anon_const_id) => SolverDefId::AnonConstId(anon_const_id),
        }
    }
}

impl<'db> From<CallableDefId> for SolverDefId<'db> {
    #[inline]
    fn from(value: CallableDefId) -> Self {
        match value {
            CallableDefId::FunctionId(id) => id.into(),
            CallableDefId::StructId(id) => id.into(),
            CallableDefId::EnumVariantId(id) => id.into(),
        }
    }
}

impl<'db> From<DefWithBodyId> for SolverDefId<'db> {
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

impl<'db> From<InferBodyId<'db>> for SolverDefId<'db> {
    #[inline]
    fn from(value: InferBodyId<'db>) -> Self {
        match value {
            InferBodyId::DefWithBodyId(id) => id.into(),
            InferBodyId::AnonConstId(id) => id.into(),
        }
    }
}

impl<'db> From<VariantId> for SolverDefId<'db> {
    #[inline]
    fn from(value: VariantId) -> Self {
        match value {
            VariantId::EnumVariantId(id) => id.into(),
            VariantId::StructId(id) => id.into(),
            VariantId::UnionId(id) => id.into(),
        }
    }
}

impl<'db> From<ExpressionStoreOwnerId> for SolverDefId<'db> {
    #[inline]
    fn from(value: ExpressionStoreOwnerId) -> Self {
        match value {
            ExpressionStoreOwnerId::Body(body_id) => body_id.into(),
            ExpressionStoreOwnerId::Signature(sig_id) => sig_id.into(),
            ExpressionStoreOwnerId::VariantFields(variant_id) => variant_id.into(),
        }
    }
}

impl TryFrom<SolverDefId<'_>> for AttrDefId {
    type Error = ();
    #[inline]
    fn try_from(value: SolverDefId<'_>) -> Result<Self, Self::Error> {
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
            SolverDefId::BuiltinDeriveImplId(_)
            | SolverDefId::InternedClosureId(_)
            | SolverDefId::InternedCoroutineId(_)
            | SolverDefId::InternedCoroutineClosureId(_)
            | SolverDefId::InternedOpaqueTyId(_)
            | SolverDefId::AnonConstId(_) => Err(()),
        }
    }
}

impl TryFrom<SolverDefId<'_>> for DefWithBodyId {
    type Error = ();

    #[inline]
    fn try_from(value: SolverDefId<'_>) -> Result<Self, Self::Error> {
        let id = match value {
            SolverDefId::ConstId(id) => id.into(),
            SolverDefId::FunctionId(id) => id.into(),
            SolverDefId::StaticId(id) => id.into(),
            SolverDefId::EnumVariantId(id) | SolverDefId::Ctor(Ctor::Enum(id)) => id.into(),
            SolverDefId::InternedOpaqueTyId(_)
            | SolverDefId::TraitId(_)
            | SolverDefId::TypeAliasId(_)
            | SolverDefId::ImplId(_)
            | SolverDefId::BuiltinDeriveImplId(_)
            | SolverDefId::InternedClosureId(_)
            | SolverDefId::InternedCoroutineId(_)
            | SolverDefId::InternedCoroutineClosureId(_)
            | SolverDefId::Ctor(Ctor::Struct(_))
            | SolverDefId::AnonConstId(_)
            | SolverDefId::AdtId(_) => return Err(()),
        };
        Ok(id)
    }
}

impl<'db> TryFrom<SolverDefId<'db>> for InferBodyId<'db> {
    type Error = ();

    #[inline]
    fn try_from(value: SolverDefId<'db>) -> Result<Self, Self::Error> {
        let id = match value {
            SolverDefId::ConstId(id) => id.into(),
            SolverDefId::FunctionId(id) => id.into(),
            SolverDefId::StaticId(id) => id.into(),
            SolverDefId::EnumVariantId(id) | SolverDefId::Ctor(Ctor::Enum(id)) => id.into(),
            SolverDefId::AnonConstId(id) => id.into(),
            SolverDefId::InternedOpaqueTyId(_)
            | SolverDefId::TraitId(_)
            | SolverDefId::TypeAliasId(_)
            | SolverDefId::ImplId(_)
            | SolverDefId::BuiltinDeriveImplId(_)
            | SolverDefId::InternedClosureId(_)
            | SolverDefId::InternedCoroutineId(_)
            | SolverDefId::InternedCoroutineClosureId(_)
            | SolverDefId::Ctor(Ctor::Struct(_))
            | SolverDefId::AdtId(_) => return Err(()),
        };
        Ok(id)
    }
}

impl TryFrom<SolverDefId<'_>> for GenericDefId {
    type Error = ();

    fn try_from(value: SolverDefId<'_>) -> Result<Self, Self::Error> {
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
            | SolverDefId::InternedCoroutineClosureId(_)
            | SolverDefId::InternedOpaqueTyId(_)
            | SolverDefId::EnumVariantId(_)
            | SolverDefId::BuiltinDeriveImplId(_)
            | SolverDefId::AnonConstId(_)
            | SolverDefId::Ctor(_) => return Err(()),
        })
    }
}

impl<'db> inherent::DefId<DbInterner<'db>> for SolverDefId<'db> {
    fn as_local(self) -> Option<SolverDefId<'db>> {
        Some(self)
    }
    fn is_local(self) -> bool {
        true
    }
}

macro_rules! declare_id_wrapper {
    ($name:ident, $wraps:ident) => {
        declare_id_wrapper!($name, $wraps, SolverDefId<'db>);
    };

    ($name:ident, $wraps:ident, $local:ty) => {
        declare_id_wrapper!($name, $wraps, $local, no_try_from);

        impl TryFrom<SolverDefId<'_>> for $name {
            type Error = ();

            #[inline]
            fn try_from(value: SolverDefId<'_>) -> Result<Self, Self::Error> {
                match value {
                    SolverDefId::$wraps(it) => Ok(Self(it)),
                    _ => Err(()),
                }
            }
        }
    };

    ($name:ident, $wraps:ident, $local:ty, no_try_from) => {
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

        impl<'db> From<$name> for SolverDefId<'db> {
            #[inline]
            fn from(value: $name) -> SolverDefId<'db> {
                value.0.into()
            }
        }

        impl<'db> inherent::DefId<DbInterner<'db>, $local> for $name {
            fn as_local(self) -> Option<$local> {
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
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClosureIdWrapper<'db>(pub InternedClosureId<'db>);

impl std::fmt::Debug for ClosureIdWrapper<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&SolverDefId::from(self.0), f)
    }
}

impl<'db> From<ClosureIdWrapper<'db>> for InternedClosureId<'db> {
    #[inline]
    fn from(value: ClosureIdWrapper<'db>) -> InternedClosureId<'db> {
        value.0
    }
}

impl<'db> From<InternedClosureId<'db>> for ClosureIdWrapper<'db> {
    #[inline]
    fn from(value: InternedClosureId<'db>) -> ClosureIdWrapper<'db> {
        Self(value)
    }
}

impl<'db> From<ClosureIdWrapper<'db>> for SolverDefId<'db> {
    #[inline]
    fn from(value: ClosureIdWrapper<'db>) -> SolverDefId<'db> {
        value.0.into()
    }
}

impl<'db> TryFrom<SolverDefId<'db>> for ClosureIdWrapper<'db> {
    type Error = ();

    #[inline]
    fn try_from(value: SolverDefId<'db>) -> Result<Self, Self::Error> {
        match value {
            SolverDefId::InternedClosureId(it) => Ok(Self(it)),
            _ => Err(()),
        }
    }
}

impl<'db> inherent::DefId<DbInterner<'db>, SolverDefId<'db>> for ClosureIdWrapper<'db> {
    fn as_local(self) -> Option<SolverDefId<'db>> {
        Some(self.into())
    }
    fn is_local(self) -> bool {
        true
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoroutineIdWrapper<'db>(pub InternedCoroutineId<'db>);

impl std::fmt::Debug for CoroutineIdWrapper<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&SolverDefId::from(self.0), f)
    }
}

impl<'db> From<CoroutineIdWrapper<'db>> for InternedCoroutineId<'db> {
    #[inline]
    fn from(value: CoroutineIdWrapper<'db>) -> InternedCoroutineId<'db> {
        value.0
    }
}

impl<'db> From<InternedCoroutineId<'db>> for CoroutineIdWrapper<'db> {
    #[inline]
    fn from(value: InternedCoroutineId<'db>) -> CoroutineIdWrapper<'db> {
        Self(value)
    }
}

impl<'db> From<CoroutineIdWrapper<'db>> for SolverDefId<'db> {
    #[inline]
    fn from(value: CoroutineIdWrapper<'db>) -> SolverDefId<'db> {
        value.0.into()
    }
}

impl<'db> TryFrom<SolverDefId<'db>> for CoroutineIdWrapper<'db> {
    type Error = ();

    #[inline]
    fn try_from(value: SolverDefId<'db>) -> Result<Self, Self::Error> {
        match value {
            SolverDefId::InternedCoroutineId(it) => Ok(Self(it)),
            _ => Err(()),
        }
    }
}

impl<'db> inherent::DefId<DbInterner<'db>, SolverDefId<'db>> for CoroutineIdWrapper<'db> {
    fn as_local(self) -> Option<SolverDefId<'db>> {
        Some(self.into())
    }
    fn is_local(self) -> bool {
        true
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoroutineClosureIdWrapper<'db>(pub InternedCoroutineClosureId<'db>);

impl std::fmt::Debug for CoroutineClosureIdWrapper<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&SolverDefId::from(self.0), f)
    }
}

impl<'db> From<CoroutineClosureIdWrapper<'db>> for InternedCoroutineClosureId<'db> {
    #[inline]
    fn from(value: CoroutineClosureIdWrapper<'db>) -> InternedCoroutineClosureId<'db> {
        value.0
    }
}

impl<'db> From<InternedCoroutineClosureId<'db>> for CoroutineClosureIdWrapper<'db> {
    #[inline]
    fn from(value: InternedCoroutineClosureId<'db>) -> CoroutineClosureIdWrapper<'db> {
        Self(value)
    }
}

impl<'db> From<CoroutineClosureIdWrapper<'db>> for SolverDefId<'db> {
    #[inline]
    fn from(value: CoroutineClosureIdWrapper<'db>) -> SolverDefId<'db> {
        value.0.into()
    }
}

impl<'db> TryFrom<SolverDefId<'db>> for CoroutineClosureIdWrapper<'db> {
    type Error = ();

    #[inline]
    fn try_from(value: SolverDefId<'db>) -> Result<Self, Self::Error> {
        match value {
            SolverDefId::InternedCoroutineClosureId(it) => Ok(Self(it)),
            _ => Err(()),
        }
    }
}

impl<'db> inherent::DefId<DbInterner<'db>, SolverDefId<'db>> for CoroutineClosureIdWrapper<'db> {
    fn as_local(self) -> Option<SolverDefId<'db>> {
        Some(self.into())
    }
    fn is_local(self) -> bool {
        true
    }
}
declare_id_wrapper!(AdtIdWrapper, AdtId);
declare_id_wrapper!(OpaqueTyIdWrapper, InternedOpaqueTyId, OpaqueTyIdWrapper);

macro_rules! declare_ty_const_pair {
    ( $ty_id_name:ident, $const_id_name:ident, $term_id_name:ident ) => {
        declare_id_wrapper!($ty_id_name, TypeAliasId);
        declare_id_wrapper!($const_id_name, ConstId);
        declare_id_wrapper!($term_id_name, TermId, SolverDefId<'db>, no_try_from);

        impl TryFrom<SolverDefId<'_>> for $term_id_name {
            type Error = ();

            #[inline]
            fn try_from(value: SolverDefId<'_>) -> Result<Self, Self::Error> {
                match value {
                    SolverDefId::TypeAliasId(it) => Ok(Self(TermId::TypeAliasId(it))),
                    SolverDefId::ConstId(it) => Ok(Self(TermId::ConstId(it))),
                    _ => Err(()),
                }
            }
        }

        impl From<$ty_id_name> for $term_id_name {
            fn from(value: $ty_id_name) -> Self {
                $term_id_name(TermId::TypeAliasId(value.0))
            }
        }

        impl From<$const_id_name> for $term_id_name {
            fn from(value: $const_id_name) -> Self {
                $term_id_name(TermId::ConstId(value.0))
            }
        }

        impl TryFrom<$term_id_name> for $ty_id_name {
            type Error = ();

            fn try_from(value: $term_id_name) -> Result<Self, Self::Error> {
                match value.0 {
                    TermId::TypeAliasId(id) => Ok($ty_id_name(id)),
                    TermId::ConstId(_) => Err(()),
                }
            }
        }

        impl TryFrom<$term_id_name> for $const_id_name {
            type Error = ();

            fn try_from(value: $term_id_name) -> Result<Self, Self::Error> {
                match value.0 {
                    TermId::ConstId(id) => Ok($const_id_name(id)),
                    TermId::TypeAliasId(_) => Err(()),
                }
            }
        }

        impl<'db> From<$const_id_name> for GeneralConstIdWrapper<'db> {
            fn from(value: $const_id_name) -> Self {
                GeneralConstIdWrapper(GeneralConstId::ConstId(value.0))
            }
        }
    };
}

declare_ty_const_pair!(TraitAssocTyId, TraitAssocConstId, TraitAssocTermId);
declare_ty_const_pair!(ImplOrTraitAssocTyId, ImplOrTraitAssocConstId, ImplOrTraitAssocTermId);
declare_ty_const_pair!(FreeTyAliasId, FreeConstAliasId, FreeTermAliasId);
declare_ty_const_pair!(InherentAssocTyId, InherentAssocConstId, InherentAssocTermId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TermId {
    TypeAliasId(TypeAliasId),
    ConstId(ConstId),
}
impl_from!(TypeAliasId, ConstId for TermId);

impl<'db> From<TermId> for SolverDefId<'db> {
    fn from(value: TermId) -> Self {
        match value {
            TermId::TypeAliasId(id) => id.into(),
            TermId::ConstId(id) => id.into(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GeneralConstIdWrapper<'db>(pub GeneralConstId<'db>);

impl std::fmt::Debug for GeneralConstIdWrapper<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}
impl<'db> From<GeneralConstIdWrapper<'db>> for GeneralConstId<'db> {
    #[inline]
    fn from(value: GeneralConstIdWrapper<'db>) -> GeneralConstId<'db> {
        value.0
    }
}
impl<'db> From<GeneralConstId<'db>> for GeneralConstIdWrapper<'db> {
    #[inline]
    fn from(value: GeneralConstId<'db>) -> GeneralConstIdWrapper<'db> {
        Self(value)
    }
}
impl<'db> From<GeneralConstIdWrapper<'db>> for SolverDefId<'db> {
    #[inline]
    fn from(value: GeneralConstIdWrapper<'db>) -> SolverDefId<'db> {
        match value.0 {
            GeneralConstId::ConstId(id) => SolverDefId::ConstId(id),
            GeneralConstId::StaticId(id) => SolverDefId::StaticId(id),
            GeneralConstId::AnonConstId(id) => SolverDefId::AnonConstId(id),
        }
    }
}
impl<'db> TryFrom<SolverDefId<'db>> for GeneralConstIdWrapper<'db> {
    type Error = ();
    #[inline]
    fn try_from(value: SolverDefId<'db>) -> Result<Self, Self::Error> {
        match value {
            SolverDefId::ConstId(it) => Ok(Self(it.into())),
            SolverDefId::StaticId(it) => Ok(Self(it.into())),
            SolverDefId::AnonConstId(it) => Ok(Self(it.into())),
            _ => Err(()),
        }
    }
}
impl<'db> inherent::DefId<DbInterner<'db>> for GeneralConstIdWrapper<'db> {
    fn as_local(self) -> Option<SolverDefId<'db>> {
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
impl<'db> From<CallableIdWrapper> for SolverDefId<'db> {
    #[inline]
    fn from(value: CallableIdWrapper) -> SolverDefId<'db> {
        match value.0 {
            CallableDefId::FunctionId(it) => it.into(),
            CallableDefId::StructId(it) => Ctor::Struct(it).into(),
            CallableDefId::EnumVariantId(it) => Ctor::Enum(it).into(),
        }
    }
}
impl TryFrom<SolverDefId<'_>> for CallableIdWrapper {
    type Error = ();
    #[inline]
    fn try_from(value: SolverDefId<'_>) -> Result<Self, Self::Error> {
        match value {
            SolverDefId::FunctionId(it) => Ok(Self(it.into())),
            SolverDefId::Ctor(Ctor::Struct(it)) => Ok(Self(it.into())),
            SolverDefId::Ctor(Ctor::Enum(it)) => Ok(Self(it.into())),
            _ => Err(()),
        }
    }
}
impl<'db> inherent::DefId<DbInterner<'db>> for CallableIdWrapper {
    fn as_local(self) -> Option<SolverDefId<'db>> {
        Some(self.into())
    }
    fn is_local(self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnyImplId {
    ImplId(ImplId),
    BuiltinDeriveImplId(BuiltinDeriveImplId),
}

impl_from!(ImplId, BuiltinDeriveImplId for AnyImplId);

impl<'db> From<AnyImplId> for SolverDefId<'db> {
    #[inline]
    fn from(value: AnyImplId) -> SolverDefId<'db> {
        match value {
            AnyImplId::ImplId(it) => it.into(),
            AnyImplId::BuiltinDeriveImplId(it) => it.into(),
        }
    }
}
impl TryFrom<SolverDefId<'_>> for AnyImplId {
    type Error = ();
    #[inline]
    fn try_from(value: SolverDefId<'_>) -> Result<Self, Self::Error> {
        match value {
            SolverDefId::ImplId(it) => Ok(it.into()),
            SolverDefId::BuiltinDeriveImplId(it) => Ok(it.into()),
            _ => Err(()),
        }
    }
}
impl<'db> inherent::DefId<DbInterner<'db>> for AnyImplId {
    fn as_local(self) -> Option<SolverDefId<'db>> {
        Some(self.into())
    }
    fn is_local(self) -> bool {
        true
    }
}
