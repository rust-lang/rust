//! Item signature IR definitions

use std::{cell::LazyCell, ops::Not as _};

use bitflags::bitflags;
use cfg::{CfgExpr, CfgOptions};
use hir_expand::{
    InFile, Intern, Lookup,
    name::{AsName, Name},
};
use intern::{Symbol, sym};
use la_arena::{Arena, Idx};
use rustc_abi::{IntegerType, ReprOptions};
use syntax::{
    AstNode, NodeOrToken, SyntaxNodePtr, T,
    ast::{self, HasGenericParams, HasName, HasVisibility, IsString},
};
use thin_vec::ThinVec;
use triomphe::Arc;

use crate::{
    ConstId, EnumId, EnumVariantId, EnumVariantLoc, ExternBlockId, FunctionId, HasModule, ImplId,
    ItemContainerId, ModuleId, StaticId, StructId, TraitId, TypeAliasId, UnionId, VariantId,
    attrs::AttrFlags,
    db::DefDatabase,
    expr_store::{
        ExpressionStore, ExpressionStoreSourceMap,
        lower::{
            ExprCollector, lower_function, lower_generic_params, lower_trait, lower_type_alias,
        },
    },
    hir::{ExprId, PatId, generics::GenericParams},
    item_tree::{FieldsShape, RawVisibility, visibility_from_ast},
    src::HasSource,
    type_ref::{TraitRef, TypeBound, TypeRefId},
};

#[inline]
fn as_name_opt(name: Option<ast::Name>) -> Name {
    name.map_or_else(Name::missing, |it| it.as_name())
}

#[derive(Debug, PartialEq, Eq)]
pub struct StructSignature {
    pub name: Name,
    pub generic_params: Arc<GenericParams>,
    pub store: Arc<ExpressionStore>,
    pub flags: StructFlags,
    pub shape: FieldsShape,
}

bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct StructFlags: u8 {
        /// Indicates whether this struct has `#[repr]`.
        const HAS_REPR = 1 << 0;
        /// Indicates whether the struct has a `#[rustc_has_incoherent_inherent_impls]` attribute.
        const RUSTC_HAS_INCOHERENT_INHERENT_IMPLS = 1 << 1;
        /// Indicates whether the struct has a `#[fundamental]` attribute.
        const FUNDAMENTAL      = 1 << 2;
        /// Indicates whether the struct is `PhantomData`.
        const IS_PHANTOM_DATA  = 1 << 3;
        /// Indicates whether this struct is `Box`.
        const IS_BOX           = 1 << 4;
        /// Indicates whether this struct is `ManuallyDrop`.
        const IS_MANUALLY_DROP = 1 << 5;
        /// Indicates whether this struct is `UnsafeCell`.
        const IS_UNSAFE_CELL   = 1 << 6;
        /// Indicates whether this struct is `UnsafePinned`.
        const IS_UNSAFE_PINNED = 1 << 7;
    }
}

impl StructSignature {
    pub fn query(db: &dyn DefDatabase, id: StructId) -> (Arc<Self>, Arc<ExpressionStoreSourceMap>) {
        let loc = id.lookup(db);
        let InFile { file_id, value: source } = loc.source(db);
        let attrs = AttrFlags::query(db, id.into());

        let mut flags = StructFlags::empty();
        if attrs.contains(AttrFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS) {
            flags |= StructFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS;
        }
        if attrs.contains(AttrFlags::FUNDAMENTAL) {
            flags |= StructFlags::FUNDAMENTAL;
        }
        if attrs.contains(AttrFlags::HAS_REPR) {
            flags |= StructFlags::HAS_REPR;
        }
        if let Some(lang) = attrs.lang_item_with_attrs(db, id.into()) {
            match lang {
                _ if lang == sym::phantom_data => flags |= StructFlags::IS_PHANTOM_DATA,
                _ if lang == sym::owned_box => flags |= StructFlags::IS_BOX,
                _ if lang == sym::manually_drop => flags |= StructFlags::IS_MANUALLY_DROP,
                _ if lang == sym::unsafe_cell => flags |= StructFlags::IS_UNSAFE_CELL,
                _ if lang == sym::unsafe_pinned => flags |= StructFlags::IS_UNSAFE_PINNED,
                _ => (),
            }
        }
        let shape = adt_shape(source.kind());

        let (store, generic_params, source_map) = lower_generic_params(
            db,
            loc.container,
            id.into(),
            file_id,
            source.generic_param_list(),
            source.where_clause(),
        );
        (
            Arc::new(StructSignature {
                generic_params,
                store,
                flags,
                shape,
                name: as_name_opt(source.name()),
            }),
            Arc::new(source_map),
        )
    }

    #[inline]
    pub fn repr(&self, db: &dyn DefDatabase, id: StructId) -> Option<ReprOptions> {
        if self.flags.contains(StructFlags::HAS_REPR) {
            AttrFlags::repr(db, id.into())
        } else {
            None
        }
    }
}

#[inline]
fn adt_shape(adt_kind: ast::StructKind) -> FieldsShape {
    match adt_kind {
        ast::StructKind::Record(_) => FieldsShape::Record,
        ast::StructKind::Tuple(_) => FieldsShape::Tuple,
        ast::StructKind::Unit => FieldsShape::Unit,
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct UnionSignature {
    pub name: Name,
    pub generic_params: Arc<GenericParams>,
    pub store: Arc<ExpressionStore>,
    pub flags: StructFlags,
}

impl UnionSignature {
    pub fn query(db: &dyn DefDatabase, id: UnionId) -> (Arc<Self>, Arc<ExpressionStoreSourceMap>) {
        let loc = id.lookup(db);
        let attrs = AttrFlags::query(db, id.into());
        let mut flags = StructFlags::empty();
        if attrs.contains(AttrFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS) {
            flags |= StructFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS;
        }
        if attrs.contains(AttrFlags::FUNDAMENTAL) {
            flags |= StructFlags::FUNDAMENTAL;
        }
        if attrs.contains(AttrFlags::HAS_REPR) {
            flags |= StructFlags::HAS_REPR;
        }

        let InFile { file_id, value: source } = loc.source(db);
        let (store, generic_params, source_map) = lower_generic_params(
            db,
            loc.container,
            id.into(),
            file_id,
            source.generic_param_list(),
            source.where_clause(),
        );
        (
            Arc::new(UnionSignature {
                generic_params,
                store,
                flags,
                name: as_name_opt(source.name()),
            }),
            Arc::new(source_map),
        )
    }
}

bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct EnumFlags: u8 {
        /// Indicates whether this enum has `#[repr]`.
        const HAS_REPR = 1 << 0;
        /// Indicates whether the enum has a `#[rustc_has_incoherent_inherent_impls]` attribute.
        const RUSTC_HAS_INCOHERENT_INHERENT_IMPLS  = 1 << 1;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct EnumSignature {
    pub name: Name,
    pub generic_params: Arc<GenericParams>,
    pub store: Arc<ExpressionStore>,
    pub flags: EnumFlags,
}

impl EnumSignature {
    pub fn query(db: &dyn DefDatabase, id: EnumId) -> (Arc<Self>, Arc<ExpressionStoreSourceMap>) {
        let loc = id.lookup(db);
        let attrs = AttrFlags::query(db, id.into());
        let mut flags = EnumFlags::empty();
        if attrs.contains(AttrFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS) {
            flags |= EnumFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS;
        }
        if attrs.contains(AttrFlags::HAS_REPR) {
            flags |= EnumFlags::HAS_REPR;
        }

        let InFile { file_id, value: source } = loc.source(db);
        let (store, generic_params, source_map) = lower_generic_params(
            db,
            loc.container,
            id.into(),
            file_id,
            source.generic_param_list(),
            source.where_clause(),
        );

        (
            Arc::new(EnumSignature {
                generic_params,
                store,
                flags,
                name: as_name_opt(source.name()),
            }),
            Arc::new(source_map),
        )
    }

    pub fn variant_body_type(db: &dyn DefDatabase, id: EnumId) -> IntegerType {
        match AttrFlags::repr(db, id.into()) {
            Some(ReprOptions { int: Some(builtin), .. }) => builtin,
            _ => IntegerType::Pointer(true),
        }
    }

    #[inline]
    pub fn repr(&self, db: &dyn DefDatabase, id: EnumId) -> Option<ReprOptions> {
        if self.flags.contains(EnumFlags::HAS_REPR) { AttrFlags::repr(db, id.into()) } else { None }
    }
}
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
    pub struct ConstFlags: u8 {
        const HAS_BODY = 1 << 1;
        const RUSTC_ALLOW_INCOHERENT_IMPL = 1 << 7;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ConstSignature {
    pub name: Option<Name>,
    // generic_params: Arc<GenericParams>,
    pub store: Arc<ExpressionStore>,
    pub type_ref: TypeRefId,
    pub flags: ConstFlags,
}

impl ConstSignature {
    pub fn query(db: &dyn DefDatabase, id: ConstId) -> (Arc<Self>, Arc<ExpressionStoreSourceMap>) {
        let loc = id.lookup(db);

        let module = loc.container.module(db);
        let attrs = AttrFlags::query(db, id.into());
        let mut flags = ConstFlags::empty();
        if attrs.contains(AttrFlags::RUSTC_ALLOW_INCOHERENT_IMPL) {
            flags |= ConstFlags::RUSTC_ALLOW_INCOHERENT_IMPL;
        }
        let source = loc.source(db);
        if source.value.body().is_some() {
            flags.insert(ConstFlags::HAS_BODY);
        }

        let (store, source_map, type_ref) =
            crate::expr_store::lower::lower_type_ref(db, module, source.as_ref().map(|it| it.ty()));

        (
            Arc::new(ConstSignature {
                store: Arc::new(store),
                type_ref,
                flags,
                name: source.value.name().map(|it| it.as_name()),
            }),
            Arc::new(source_map),
        )
    }

    pub fn has_body(&self) -> bool {
        self.flags.contains(ConstFlags::HAS_BODY)
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
    pub struct StaticFlags: u8 {
        const HAS_BODY = 1 << 1;
        const MUTABLE    = 1 << 3;
        const UNSAFE     = 1 << 4;
        const EXPLICIT_SAFE = 1 << 5;
        const EXTERN     = 1 << 6;
        const RUSTC_ALLOW_INCOHERENT_IMPL = 1 << 7;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct StaticSignature {
    pub name: Name,

    // generic_params: Arc<GenericParams>,
    pub store: Arc<ExpressionStore>,
    pub type_ref: TypeRefId,
    pub flags: StaticFlags,
}
impl StaticSignature {
    pub fn query(db: &dyn DefDatabase, id: StaticId) -> (Arc<Self>, Arc<ExpressionStoreSourceMap>) {
        let loc = id.lookup(db);

        let module = loc.container.module(db);
        let attrs = AttrFlags::query(db, id.into());
        let mut flags = StaticFlags::empty();
        if attrs.contains(AttrFlags::RUSTC_ALLOW_INCOHERENT_IMPL) {
            flags |= StaticFlags::RUSTC_ALLOW_INCOHERENT_IMPL;
        }

        if matches!(loc.container, ItemContainerId::ExternBlockId(_)) {
            flags.insert(StaticFlags::EXTERN);
        }

        let source = loc.source(db);
        if source.value.body().is_some() {
            flags.insert(StaticFlags::HAS_BODY);
        }
        if source.value.mut_token().is_some() {
            flags.insert(StaticFlags::MUTABLE);
        }
        if source.value.unsafe_token().is_some() {
            flags.insert(StaticFlags::UNSAFE);
        }
        if source.value.safe_token().is_some() {
            flags.insert(StaticFlags::EXPLICIT_SAFE);
        }

        let (store, source_map, type_ref) =
            crate::expr_store::lower::lower_type_ref(db, module, source.as_ref().map(|it| it.ty()));

        (
            Arc::new(StaticSignature {
                store: Arc::new(store),
                type_ref,
                flags,
                name: as_name_opt(source.value.name()),
            }),
            Arc::new(source_map),
        )
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
    pub struct ImplFlags: u8 {
        const NEGATIVE = 1 << 1;
        const DEFAULT = 1 << 2;
        const UNSAFE = 1 << 3;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ImplSignature {
    pub generic_params: Arc<GenericParams>,
    pub store: Arc<ExpressionStore>,
    pub self_ty: TypeRefId,
    pub target_trait: Option<TraitRef>,
    pub flags: ImplFlags,
}

impl ImplSignature {
    pub fn query(db: &dyn DefDatabase, id: ImplId) -> (Arc<Self>, Arc<ExpressionStoreSourceMap>) {
        let loc = id.lookup(db);

        let mut flags = ImplFlags::empty();
        let src = loc.source(db);
        if src.value.unsafe_token().is_some() {
            flags.insert(ImplFlags::UNSAFE);
        }
        if src.value.excl_token().is_some() {
            flags.insert(ImplFlags::NEGATIVE);
        }
        if src.value.default_token().is_some() {
            flags.insert(ImplFlags::DEFAULT);
        }

        let (store, source_map, self_ty, target_trait, generic_params) =
            crate::expr_store::lower::lower_impl(db, loc.container, src, id);

        (
            Arc::new(ImplSignature {
                store: Arc::new(store),
                generic_params,
                self_ty,
                target_trait,
                flags,
            }),
            Arc::new(source_map),
        )
    }

    #[inline]
    pub fn is_negative(&self) -> bool {
        self.flags.contains(ImplFlags::NEGATIVE)
    }

    #[inline]
    pub fn is_default(&self) -> bool {
        self.flags.contains(ImplFlags::DEFAULT)
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
    pub struct TraitFlags: u16 {
        const RUSTC_HAS_INCOHERENT_INHERENT_IMPLS = 1 << 1;
        const FUNDAMENTAL = 1 << 2;
        const UNSAFE = 1 << 3;
        const AUTO = 1 << 4;
        const SKIP_ARRAY_DURING_METHOD_DISPATCH = 1 << 5;
        const SKIP_BOXED_SLICE_DURING_METHOD_DISPATCH = 1 << 6;
        const RUSTC_PAREN_SUGAR = 1 << 7;
        const COINDUCTIVE = 1 << 8;
        const ALIAS = 1 << 9;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct TraitSignature {
    pub name: Name,
    pub generic_params: Arc<GenericParams>,
    pub store: Arc<ExpressionStore>,
    pub flags: TraitFlags,
}

impl TraitSignature {
    pub fn query(db: &dyn DefDatabase, id: TraitId) -> (Arc<Self>, Arc<ExpressionStoreSourceMap>) {
        let loc = id.lookup(db);

        let mut flags = TraitFlags::empty();
        let attrs = AttrFlags::query(db, id.into());
        let source = loc.source(db);
        if source.value.auto_token().is_some() {
            flags.insert(TraitFlags::AUTO);
        }
        if source.value.unsafe_token().is_some() {
            flags.insert(TraitFlags::UNSAFE);
        }
        if source.value.eq_token().is_some() {
            flags.insert(TraitFlags::ALIAS);
        }
        if attrs.contains(AttrFlags::FUNDAMENTAL) {
            flags |= TraitFlags::FUNDAMENTAL;
        }
        if attrs.contains(AttrFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS) {
            flags |= TraitFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS;
        }
        if attrs.contains(AttrFlags::RUSTC_PAREN_SUGAR) {
            flags |= TraitFlags::RUSTC_PAREN_SUGAR;
        }
        if attrs.contains(AttrFlags::RUSTC_COINDUCTIVE) {
            flags |= TraitFlags::COINDUCTIVE;
        }

        if attrs.contains(AttrFlags::RUSTC_SKIP_ARRAY_DURING_METHOD_DISPATCH) {
            flags |= TraitFlags::SKIP_ARRAY_DURING_METHOD_DISPATCH;
        }
        if attrs.contains(AttrFlags::RUSTC_SKIP_BOXED_SLICE_DURING_METHOD_DISPATCH) {
            flags |= TraitFlags::SKIP_BOXED_SLICE_DURING_METHOD_DISPATCH;
        }

        let name = as_name_opt(source.value.name());
        let (store, source_map, generic_params) = lower_trait(db, loc.container, source, id);

        (
            Arc::new(TraitSignature { store: Arc::new(store), generic_params, flags, name }),
            Arc::new(source_map),
        )
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
    pub struct FnFlags: u16 {
        const HAS_BODY = 1 << 1;
        const DEFAULT = 1 << 2;
        const CONST = 1 << 3;
        const ASYNC = 1 << 4;
        const UNSAFE = 1 << 5;
        const HAS_VARARGS = 1 << 6;
        const RUSTC_ALLOW_INCOHERENT_IMPL = 1 << 7;
        const HAS_SELF_PARAM = 1 << 8;
        /// The `#[target_feature]` attribute is necessary to check safety (with RFC 2396),
        /// but keeping it for all functions will consume a lot of memory when there are
        /// only very few functions with it. So we only encode its existence here, and lookup
        /// it if needed.
        const HAS_TARGET_FEATURE = 1 << 9;
        const DEPRECATED_SAFE_2024 = 1 << 10;
        const EXPLICIT_SAFE = 1 << 11;
        const HAS_LEGACY_CONST_GENERICS = 1 << 12;
        const RUSTC_INTRINSIC = 1 << 13;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct FunctionSignature {
    pub name: Name,
    pub generic_params: Arc<GenericParams>,
    pub store: Arc<ExpressionStore>,
    pub params: Box<[TypeRefId]>,
    pub ret_type: Option<TypeRefId>,
    pub abi: Option<Symbol>,
    pub flags: FnFlags,
}

impl FunctionSignature {
    pub fn query(
        db: &dyn DefDatabase,
        id: FunctionId,
    ) -> (Arc<Self>, Arc<ExpressionStoreSourceMap>) {
        let loc = id.lookup(db);
        let module = loc.container.module(db);

        let mut flags = FnFlags::empty();
        let attrs = AttrFlags::query(db, id.into());
        if attrs.contains(AttrFlags::RUSTC_ALLOW_INCOHERENT_IMPL) {
            flags.insert(FnFlags::RUSTC_ALLOW_INCOHERENT_IMPL);
        }

        if attrs.contains(AttrFlags::HAS_TARGET_FEATURE) {
            flags.insert(FnFlags::HAS_TARGET_FEATURE);
        }

        if attrs.contains(AttrFlags::RUSTC_INTRINSIC) {
            flags.insert(FnFlags::RUSTC_INTRINSIC);
        }
        if attrs.contains(AttrFlags::HAS_LEGACY_CONST_GENERICS) {
            flags.insert(FnFlags::HAS_LEGACY_CONST_GENERICS);
        }

        let source = loc.source(db);

        if source.value.unsafe_token().is_some() {
            if attrs.contains(AttrFlags::RUSTC_DEPRECATED_SAFE_2024) {
                flags.insert(FnFlags::DEPRECATED_SAFE_2024);
            } else {
                flags.insert(FnFlags::UNSAFE);
            }
        }
        if source.value.async_token().is_some() {
            flags.insert(FnFlags::ASYNC);
        }
        if source.value.const_token().is_some() {
            flags.insert(FnFlags::CONST);
        }
        if source.value.default_token().is_some() {
            flags.insert(FnFlags::DEFAULT);
        }
        if source.value.safe_token().is_some() {
            flags.insert(FnFlags::EXPLICIT_SAFE);
        }
        if source.value.body().is_some() {
            flags.insert(FnFlags::HAS_BODY);
        }

        let name = as_name_opt(source.value.name());
        let abi = source.value.abi().map(|abi| {
            abi.abi_string().map_or_else(|| sym::C, |it| Symbol::intern(it.text_without_quotes()))
        });
        let (store, source_map, generic_params, params, ret_type, self_param, variadic) =
            lower_function(db, module, source, id);
        if self_param {
            flags.insert(FnFlags::HAS_SELF_PARAM);
        }
        if variadic {
            flags.insert(FnFlags::HAS_VARARGS);
        }
        (
            Arc::new(FunctionSignature {
                generic_params,
                store: Arc::new(store),
                params,
                ret_type,
                abi,
                flags,
                name,
            }),
            Arc::new(source_map),
        )
    }

    pub fn has_body(&self) -> bool {
        self.flags.contains(FnFlags::HAS_BODY)
    }

    /// True if the first param is `self`. This is relevant to decide whether this
    /// can be called as a method.
    pub fn has_self_param(&self) -> bool {
        self.flags.contains(FnFlags::HAS_SELF_PARAM)
    }

    pub fn is_default(&self) -> bool {
        self.flags.contains(FnFlags::DEFAULT)
    }

    pub fn is_const(&self) -> bool {
        self.flags.contains(FnFlags::CONST)
    }

    pub fn is_async(&self) -> bool {
        self.flags.contains(FnFlags::ASYNC)
    }

    pub fn is_unsafe(&self) -> bool {
        self.flags.contains(FnFlags::UNSAFE)
    }

    pub fn is_deprecated_safe_2024(&self) -> bool {
        self.flags.contains(FnFlags::DEPRECATED_SAFE_2024)
    }

    pub fn is_safe(&self) -> bool {
        self.flags.contains(FnFlags::EXPLICIT_SAFE)
    }

    pub fn is_varargs(&self) -> bool {
        self.flags.contains(FnFlags::HAS_VARARGS)
    }

    pub fn has_target_feature(&self) -> bool {
        self.flags.contains(FnFlags::HAS_TARGET_FEATURE)
    }

    #[inline]
    pub fn legacy_const_generics_indices<'db>(
        &self,
        db: &'db dyn DefDatabase,
        id: FunctionId,
    ) -> Option<&'db [u32]> {
        if !self.flags.contains(FnFlags::HAS_LEGACY_CONST_GENERICS) {
            return None;
        }

        AttrFlags::legacy_const_generic_indices(db, id).as_deref()
    }

    pub fn is_intrinsic(db: &dyn DefDatabase, id: FunctionId) -> bool {
        let data = db.function_signature(id);
        data.flags.contains(FnFlags::RUSTC_INTRINSIC)
            // Keep this around for a bit until extern "rustc-intrinsic" abis are no longer used
            || match &data.abi {
                Some(abi) => *abi == sym::rust_dash_intrinsic,
                None => match id.lookup(db).container {
                    ItemContainerId::ExternBlockId(block) => {
                        block.abi(db) == Some(sym::rust_dash_intrinsic)
                    }
                    _ => false,
                },
            }
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
    pub struct TypeAliasFlags: u8 {
        const RUSTC_HAS_INCOHERENT_INHERENT_IMPL = 1 << 1;
        const IS_EXTERN = 1 << 6;
        const RUSTC_ALLOW_INCOHERENT_IMPL = 1 << 7;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct TypeAliasSignature {
    pub name: Name,
    pub generic_params: Arc<GenericParams>,
    pub store: Arc<ExpressionStore>,
    pub bounds: Box<[TypeBound]>,
    pub ty: Option<TypeRefId>,
    pub flags: TypeAliasFlags,
}

impl TypeAliasSignature {
    pub fn query(
        db: &dyn DefDatabase,
        id: TypeAliasId,
    ) -> (Arc<Self>, Arc<ExpressionStoreSourceMap>) {
        let loc = id.lookup(db);

        let mut flags = TypeAliasFlags::empty();
        let attrs = AttrFlags::query(db, id.into());
        if attrs.contains(AttrFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS) {
            flags.insert(TypeAliasFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPL);
        }
        if attrs.contains(AttrFlags::RUSTC_ALLOW_INCOHERENT_IMPL) {
            flags.insert(TypeAliasFlags::RUSTC_ALLOW_INCOHERENT_IMPL);
        }
        if matches!(loc.container, ItemContainerId::ExternBlockId(_)) {
            flags.insert(TypeAliasFlags::IS_EXTERN);
        }
        let source = loc.source(db);
        let name = as_name_opt(source.value.name());
        let (store, source_map, generic_params, bounds, ty) =
            lower_type_alias(db, loc.container.module(db), source, id);

        (
            Arc::new(TypeAliasSignature {
                store: Arc::new(store),
                generic_params,
                flags,
                bounds,
                name,
                ty,
            }),
            Arc::new(source_map),
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct FunctionBody {
    pub store: Arc<ExpressionStore>,
    pub parameters: Box<[PatId]>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct SimpleBody {
    pub store: Arc<ExpressionStore>,
}
pub type StaticBody = SimpleBody;
pub type ConstBody = SimpleBody;
pub type EnumVariantBody = SimpleBody;

#[derive(Debug, PartialEq, Eq)]
pub struct VariantFieldsBody {
    pub store: Arc<ExpressionStore>,
    pub fields: Box<[Option<ExprId>]>,
}

/// A single field of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldData {
    pub name: Name,
    pub type_ref: TypeRefId,
    pub visibility: RawVisibility,
    pub is_unsafe: bool,
    pub default_value: Option<ExprId>,
}

pub type LocalFieldId = Idx<FieldData>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantFields {
    fields: Arena<FieldData>,
    pub store: Arc<ExpressionStore>,
    pub shape: FieldsShape,
}

#[salsa::tracked]
impl VariantFields {
    #[salsa::tracked(returns(clone))]
    pub(crate) fn query(
        db: &dyn DefDatabase,
        id: VariantId,
    ) -> (Arc<Self>, Arc<ExpressionStoreSourceMap>) {
        let (shape, result) = match id {
            VariantId::EnumVariantId(id) => {
                let loc = id.lookup(db);
                let parent = loc.parent.lookup(db);
                let source = loc.source(db);
                let shape = adt_shape(source.value.kind());
                let enum_vis = Some(source.value.parent_enum().visibility());
                let fields = lower_field_list(
                    db,
                    parent.container,
                    source.map(|src| src.field_list()),
                    enum_vis,
                );
                (shape, fields)
            }
            VariantId::StructId(id) => {
                let loc = id.lookup(db);
                let source = loc.source(db);
                let shape = adt_shape(source.value.kind());
                let fields =
                    lower_field_list(db, loc.container, source.map(|src| src.field_list()), None);
                (shape, fields)
            }
            VariantId::UnionId(id) => {
                let loc = id.lookup(db);
                let source = loc.source(db);
                let fields = lower_field_list(
                    db,
                    loc.container,
                    source.map(|src| src.record_field_list().map(ast::FieldList::RecordFieldList)),
                    None,
                );
                (FieldsShape::Record, fields)
            }
        };
        match result {
            Some((fields, store, source_map)) => (
                Arc::new(VariantFields { fields, store: Arc::new(store), shape }),
                Arc::new(source_map),
            ),
            None => {
                let (store, source_map) = ExpressionStore::empty_singleton();
                (Arc::new(VariantFields { fields: Arena::default(), store, shape }), source_map)
            }
        }
    }

    #[salsa::tracked(returns(deref))]
    pub(crate) fn firewall(db: &dyn DefDatabase, id: VariantId) -> Arc<Self> {
        Self::query(db, id).0
    }
}

impl VariantFields {
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn fields(&self) -> &Arena<FieldData> {
        &self.fields
    }

    pub fn field(&self, name: &Name) -> Option<LocalFieldId> {
        self.fields().iter().find_map(|(id, data)| if &data.name == name { Some(id) } else { None })
    }
}

fn lower_field_list(
    db: &dyn DefDatabase,
    module: ModuleId,
    fields: InFile<Option<ast::FieldList>>,
    override_visibility: Option<Option<ast::Visibility>>,
) -> Option<(Arena<FieldData>, ExpressionStore, ExpressionStoreSourceMap)> {
    let file_id = fields.file_id;
    match fields.value? {
        ast::FieldList::RecordFieldList(fields) => lower_fields(
            db,
            module,
            InFile::new(file_id, fields.fields().map(|field| (field.ty(), field))),
            |_, field| as_name_opt(field.name()),
            override_visibility,
        ),
        ast::FieldList::TupleFieldList(fields) => lower_fields(
            db,
            module,
            InFile::new(file_id, fields.fields().map(|field| (field.ty(), field))),
            |idx, _| Name::new_tuple_field(idx),
            override_visibility,
        ),
    }
}

fn lower_fields<Field: ast::HasAttrs + ast::HasVisibility>(
    db: &dyn DefDatabase,
    module: ModuleId,
    fields: InFile<impl Iterator<Item = (Option<ast::Type>, Field)>>,
    mut field_name: impl FnMut(usize, &Field) -> Name,
    override_visibility: Option<Option<ast::Visibility>>,
) -> Option<(Arena<FieldData>, ExpressionStore, ExpressionStoreSourceMap)> {
    let cfg_options = module.krate(db).cfg_options(db);
    let mut col = ExprCollector::new(db, module, fields.file_id);
    let override_visibility = override_visibility.map(|vis| {
        LazyCell::new(|| {
            let span_map = db.span_map(fields.file_id);
            visibility_from_ast(db, vis, &mut |range| span_map.span_for_range(range).ctx)
        })
    });

    let mut arena = Arena::new();
    let mut idx = 0;
    let mut has_fields = false;
    for (ty, field) in fields.value {
        has_fields = true;
        match AttrFlags::is_cfg_enabled_for(&field, cfg_options) {
            Ok(()) => {
                let type_ref =
                    col.lower_type_ref_opt(ty, &mut ExprCollector::impl_trait_error_allocator);
                let visibility = override_visibility.as_ref().map_or_else(
                    || {
                        visibility_from_ast(db, field.visibility(), &mut |range| {
                            col.span_map().span_for_range(range).ctx
                        })
                    },
                    |it| RawVisibility::clone(it),
                );
                let is_unsafe = field
                    .syntax()
                    .children_with_tokens()
                    .filter_map(NodeOrToken::into_token)
                    .any(|token| token.kind() == T![unsafe]);
                let name = field_name(idx, &field);

                // Check if field has default value (only for record fields)
                let default_value = ast::RecordField::cast(field.syntax().clone())
                    .and_then(|rf| rf.eq_token().is_some().then_some(rf.expr()))
                    .flatten()
                    .map(|expr| col.collect_expr_opt(Some(expr)));

                arena.alloc(FieldData { name, type_ref, visibility, is_unsafe, default_value });
                idx += 1;
            }
            Err(cfg) => {
                col.store.diagnostics.push(
                    crate::expr_store::ExpressionStoreDiagnostics::InactiveCode {
                        node: InFile::new(fields.file_id, SyntaxNodePtr::new(field.syntax())),
                        cfg,
                        opts: cfg_options.clone(),
                    },
                );
            }
        }
    }
    if !has_fields {
        return None;
    }
    let (store, source_map) = col.store.finish();
    arena.shrink_to_fit();
    Some((arena, store, source_map))
}

#[derive(Debug, PartialEq, Eq)]
pub struct InactiveEnumVariantCode {
    pub cfg: CfgExpr,
    pub opts: CfgOptions,
    pub ast_id: span::FileAstId<ast::Variant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumVariants {
    pub variants: Box<[(EnumVariantId, Name, FieldsShape)]>,
}

#[salsa::tracked]
impl EnumVariants {
    #[salsa::tracked(returns(ref))]
    pub(crate) fn of(
        db: &dyn DefDatabase,
        e: EnumId,
    ) -> (EnumVariants, Option<ThinVec<InactiveEnumVariantCode>>) {
        let loc = e.lookup(db);
        let source = loc.source(db);
        let ast_id_map = db.ast_id_map(source.file_id);

        let mut diagnostics = ThinVec::new();
        let cfg_options = loc.container.krate(db).cfg_options(db);
        let mut index = 0;
        let Some(variants) = source.value.variant_list() else {
            return (EnumVariants { variants: Box::default() }, None);
        };
        let variants = variants
            .variants()
            .filter_map(|variant| {
                let ast_id = ast_id_map.ast_id(&variant);
                match AttrFlags::is_cfg_enabled_for(&variant, cfg_options) {
                    Ok(()) => {
                        let enum_variant =
                            EnumVariantLoc { id: source.with_value(ast_id), parent: e, index }
                                .intern(db);
                        index += 1;
                        let name = as_name_opt(variant.name());
                        let shape = adt_shape(variant.kind());
                        Some((enum_variant, name, shape))
                    }
                    Err(cfg) => {
                        diagnostics.push(InactiveEnumVariantCode {
                            ast_id,
                            cfg,
                            opts: cfg_options.clone(),
                        });
                        None
                    }
                }
            })
            .collect();

        (EnumVariants { variants }, diagnostics.is_empty().not().then_some(diagnostics))
    }
}

impl EnumVariants {
    pub fn variant(&self, name: &Name) -> Option<EnumVariantId> {
        self.variants.iter().find_map(|(v, n, _)| if n == name { Some(*v) } else { None })
    }

    pub fn variant_name_by_id(&self, variant_id: EnumVariantId) -> Option<Name> {
        self.variants
            .iter()
            .find_map(|(id, name, _)| if *id == variant_id { Some(name.clone()) } else { None })
    }

    // [Adopted from rustc](https://github.com/rust-lang/rust/blob/bd53aa3bf7a24a70d763182303bd75e5fc51a9af/compiler/rustc_middle/src/ty/adt.rs#L446-L448)
    pub fn is_payload_free(&self, db: &dyn DefDatabase) -> bool {
        self.variants.iter().all(|&(v, _, _)| {
            // The condition check order is slightly modified from rustc
            // to improve performance by early returning with relatively fast checks
            let variant = v.fields(db);
            if !variant.fields().is_empty() {
                return false;
            }
            // The outer if condition is whether this variant has const ctor or not
            if !matches!(variant.shape, FieldsShape::Unit) {
                let body = db.body(v.into());
                // A variant with explicit discriminant
                if !matches!(body[body.body_expr], crate::hir::Expr::Missing) {
                    return false;
                }
            }
            true
        })
    }
}

pub(crate) fn extern_block_abi(
    db: &dyn DefDatabase,
    extern_block: ExternBlockId,
) -> Option<Symbol> {
    let source = extern_block.lookup(db).source(db);
    source.value.abi().map(|abi| {
        match abi.abi_string() {
            Some(tok) => Symbol::intern(tok.text_without_quotes()),
            // `extern` default to be `extern "C"`.
            _ => sym::C,
        }
    })
}
