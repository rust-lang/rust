//! Item signature IR definitions

use std::ops::Not as _;

use bitflags::bitflags;
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{InFile, Intern, Lookup, name::Name};
use intern::{Symbol, sym};
use la_arena::{Arena, Idx};
use rustc_abi::{IntegerType, ReprOptions};
use syntax::{
    AstNode, SyntaxNodePtr,
    ast::{self, HasGenericParams, IsString},
};
use thin_vec::ThinVec;
use triomphe::Arc;

use crate::{
    ConstId, EnumId, EnumVariantId, EnumVariantLoc, FunctionId, HasModule, ImplId, ItemContainerId,
    ModuleId, StaticId, StructId, TraitAliasId, TraitId, TypeAliasId, UnionId, VariantId,
    db::DefDatabase,
    expr_store::{
        ExpressionStore, ExpressionStoreSourceMap,
        lower::{
            ExprCollector, lower_function, lower_generic_params, lower_trait, lower_trait_alias,
            lower_type_alias,
        },
    },
    hir::{ExprId, PatId, generics::GenericParams},
    item_tree::{
        AttrOwner, Field, FieldParent, FieldsShape, FileItemTreeId, ItemTree, ItemTreeId, ModItem,
        RawVisibility, RawVisibilityId,
    },
    lang_item::LangItem,
    src::HasSource,
    type_ref::{TraitRef, TypeBound, TypeRefId},
};

#[derive(Debug, PartialEq, Eq)]
pub struct StructSignature {
    pub name: Name,
    pub generic_params: Arc<GenericParams>,
    pub store: Arc<ExpressionStore>,
    pub flags: StructFlags,
    pub shape: FieldsShape,
    pub repr: Option<ReprOptions>,
}

bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct StructFlags: u8 {
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
        let item_tree = loc.id.item_tree(db);
        let attrs = item_tree.attrs(db, loc.container.krate, ModItem::from(loc.id.value).into());

        let mut flags = StructFlags::empty();
        if attrs.by_key(sym::rustc_has_incoherent_inherent_impls).exists() {
            flags |= StructFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS;
        }
        if attrs.by_key(sym::fundamental).exists() {
            flags |= StructFlags::FUNDAMENTAL;
        }
        if let Some(lang) = attrs.lang_item() {
            match lang {
                LangItem::PhantomData => flags |= StructFlags::IS_PHANTOM_DATA,
                LangItem::OwnedBox => flags |= StructFlags::IS_BOX,
                LangItem::ManuallyDrop => flags |= StructFlags::IS_MANUALLY_DROP,
                LangItem::UnsafeCell => flags |= StructFlags::IS_UNSAFE_CELL,
                LangItem::UnsafePinned => flags |= StructFlags::IS_UNSAFE_PINNED,
                _ => (),
            }
        }
        let repr = attrs.repr();

        let hir_expand::files::InFileWrapper { file_id, value } = loc.source(db);
        let (store, generic_params, source_map) = lower_generic_params(
            db,
            loc.container,
            id.into(),
            file_id,
            value.generic_param_list(),
            value.where_clause(),
        );
        (
            Arc::new(StructSignature {
                generic_params,
                store,
                flags,
                shape: item_tree[loc.id.value].shape,
                name: item_tree[loc.id.value].name.clone(),
                repr,
            }),
            Arc::new(source_map),
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct UnionSignature {
    pub name: Name,
    pub generic_params: Arc<GenericParams>,
    pub store: Arc<ExpressionStore>,
    pub flags: StructFlags,
    pub repr: Option<ReprOptions>,
}

impl UnionSignature {
    pub fn query(db: &dyn DefDatabase, id: UnionId) -> (Arc<Self>, Arc<ExpressionStoreSourceMap>) {
        let loc = id.lookup(db);
        let krate = loc.container.krate;
        let item_tree = loc.id.item_tree(db);
        let attrs = item_tree.attrs(db, krate, ModItem::from(loc.id.value).into());
        let mut flags = StructFlags::empty();
        if attrs.by_key(sym::rustc_has_incoherent_inherent_impls).exists() {
            flags |= StructFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS;
        }
        if attrs.by_key(sym::fundamental).exists() {
            flags |= StructFlags::FUNDAMENTAL;
        }

        let repr = attrs.repr();

        let hir_expand::files::InFileWrapper { file_id, value } = loc.source(db);
        let (store, generic_params, source_map) = lower_generic_params(
            db,
            loc.container,
            id.into(),
            file_id,
            value.generic_param_list(),
            value.where_clause(),
        );
        (
            Arc::new(UnionSignature {
                generic_params,
                store,
                flags,
                repr,
                name: item_tree[loc.id.value].name.clone(),
            }),
            Arc::new(source_map),
        )
    }
}

bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct EnumFlags: u8 {
        const RUSTC_HAS_INCOHERENT_INHERENT_IMPLS  = 1 << 1;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct EnumSignature {
    pub name: Name,
    pub generic_params: Arc<GenericParams>,
    pub store: Arc<ExpressionStore>,
    pub flags: EnumFlags,
    pub repr: Option<ReprOptions>,
}

impl EnumSignature {
    pub fn query(db: &dyn DefDatabase, id: EnumId) -> (Arc<Self>, Arc<ExpressionStoreSourceMap>) {
        let loc = id.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let attrs = item_tree.attrs(db, loc.container.krate, ModItem::from(loc.id.value).into());
        let mut flags = EnumFlags::empty();
        if attrs.by_key(sym::rustc_has_incoherent_inherent_impls).exists() {
            flags |= EnumFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS;
        }

        let repr = attrs.repr();

        let hir_expand::files::InFileWrapper { file_id, value } = loc.source(db);
        let (store, generic_params, source_map) = lower_generic_params(
            db,
            loc.container,
            id.into(),
            file_id,
            value.generic_param_list(),
            value.where_clause(),
        );

        (
            Arc::new(EnumSignature {
                generic_params,
                store,
                flags,
                repr,
                name: item_tree[loc.id.value].name.clone(),
            }),
            Arc::new(source_map),
        )
    }

    pub fn variant_body_type(&self) -> IntegerType {
        match self.repr {
            Some(ReprOptions { int: Some(builtin), .. }) => builtin,
            _ => IntegerType::Pointer(true),
        }
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
        let item_tree = loc.id.item_tree(db);

        let module = loc.container.module(db);
        let attrs = item_tree.attrs(db, module.krate, ModItem::from(loc.id.value).into());
        let mut flags = ConstFlags::empty();
        if attrs.by_key(sym::rustc_allow_incoherent_impl).exists() {
            flags |= ConstFlags::RUSTC_ALLOW_INCOHERENT_IMPL;
        }
        let source = loc.source(db);
        if source.value.body().is_some() {
            flags.insert(ConstFlags::HAS_BODY);
        }

        let (store, source_map, type_ref) =
            crate::expr_store::lower::lower_type_ref(db, module, source.map(|it| it.ty()));

        (
            Arc::new(ConstSignature {
                store: Arc::new(store),
                type_ref,
                flags,
                name: item_tree[loc.id.value].name.clone(),
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
        let item_tree = loc.id.item_tree(db);

        let module = loc.container.module(db);
        let attrs = item_tree.attrs(db, module.krate, ModItem::from(loc.id.value).into());
        let mut flags = StaticFlags::empty();
        if attrs.by_key(sym::rustc_allow_incoherent_impl).exists() {
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
            crate::expr_store::lower::lower_type_ref(db, module, source.map(|it| it.ty()));

        (
            Arc::new(StaticSignature {
                store: Arc::new(store),
                type_ref,
                flags,
                name: item_tree[loc.id.value].name.clone(),
            }),
            Arc::new(source_map),
        )
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
    pub struct ImplFlags: u8 {
        const NEGATIVE = 1 << 1;
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
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
    pub struct TraitFlags: u8 {
        const RUSTC_HAS_INCOHERENT_INHERENT_IMPLS = 1 << 1;
        const FUNDAMENTAL = 1 << 2;
        const UNSAFE = 1 << 3;
        const AUTO = 1 << 4;
        const SKIP_ARRAY_DURING_METHOD_DISPATCH = 1 << 5;
        const SKIP_BOXED_SLICE_DURING_METHOD_DISPATCH = 1 << 6;
        const RUSTC_PAREN_SUGAR = 1 << 7;
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
        let item_tree = loc.id.item_tree(db);

        let mut flags = TraitFlags::empty();
        let attrs = item_tree.attrs(db, loc.container.krate, ModItem::from(loc.id.value).into());
        let source = loc.source(db);
        if source.value.auto_token().is_some() {
            flags.insert(TraitFlags::AUTO);
        }
        if source.value.unsafe_token().is_some() {
            flags.insert(TraitFlags::UNSAFE);
        }
        if attrs.by_key(sym::fundamental).exists() {
            flags |= TraitFlags::FUNDAMENTAL;
        }
        if attrs.by_key(sym::rustc_has_incoherent_inherent_impls).exists() {
            flags |= TraitFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS;
        }
        if attrs.by_key(sym::rustc_paren_sugar).exists() {
            flags |= TraitFlags::RUSTC_PAREN_SUGAR;
        }
        let mut skip_array_during_method_dispatch =
            attrs.by_key(sym::rustc_skip_array_during_method_dispatch).exists();
        let mut skip_boxed_slice_during_method_dispatch = false;
        for tt in attrs.by_key(sym::rustc_skip_during_method_dispatch).tt_values() {
            for tt in tt.iter() {
                if let tt::iter::TtElement::Leaf(tt::Leaf::Ident(ident)) = tt {
                    skip_array_during_method_dispatch |= ident.sym == sym::array;
                    skip_boxed_slice_during_method_dispatch |= ident.sym == sym::boxed_slice;
                }
            }
        }

        if skip_array_during_method_dispatch {
            flags |= TraitFlags::SKIP_ARRAY_DURING_METHOD_DISPATCH;
        }
        if skip_boxed_slice_during_method_dispatch {
            flags |= TraitFlags::SKIP_BOXED_SLICE_DURING_METHOD_DISPATCH;
        }

        let (store, source_map, generic_params) = lower_trait(db, loc.container, source, id);

        (
            Arc::new(TraitSignature {
                store: Arc::new(store),
                generic_params,
                flags,
                name: item_tree[loc.id.value].name.clone(),
            }),
            Arc::new(source_map),
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct TraitAliasSignature {
    pub name: Name,
    pub generic_params: Arc<GenericParams>,
    pub store: Arc<ExpressionStore>,
}

impl TraitAliasSignature {
    pub fn query(
        db: &dyn DefDatabase,
        id: TraitAliasId,
    ) -> (Arc<Self>, Arc<ExpressionStoreSourceMap>) {
        let loc = id.lookup(db);
        let item_tree = loc.id.item_tree(db);

        let source = loc.source(db);
        let (store, source_map, generic_params) = lower_trait_alias(db, loc.container, source, id);

        (
            Arc::new(TraitAliasSignature {
                generic_params,
                store: Arc::new(store),
                name: item_tree[loc.id.value].name.clone(),
            }),
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
    // FIXME: we should put this behind a fn flags + query to avoid bloating the struct
    pub legacy_const_generics_indices: Option<Box<Box<[u32]>>>,
}

impl FunctionSignature {
    pub fn query(
        db: &dyn DefDatabase,
        id: FunctionId,
    ) -> (Arc<Self>, Arc<ExpressionStoreSourceMap>) {
        let loc = id.lookup(db);
        let module = loc.container.module(db);
        let item_tree = loc.id.item_tree(db);

        let mut flags = FnFlags::empty();
        let attrs = item_tree.attrs(db, module.krate, ModItem::from(loc.id.value).into());
        if attrs.by_key(sym::rustc_allow_incoherent_impl).exists() {
            flags.insert(FnFlags::RUSTC_ALLOW_INCOHERENT_IMPL);
        }

        if attrs.by_key(sym::target_feature).exists() {
            flags.insert(FnFlags::HAS_TARGET_FEATURE);
        }
        let legacy_const_generics_indices = attrs.rustc_legacy_const_generics();

        let source = loc.source(db);

        if source.value.unsafe_token().is_some() {
            if attrs.by_key(sym::rustc_deprecated_safe_2024).exists() {
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
                legacy_const_generics_indices,
                name: item_tree[loc.id.value].name.clone(),
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
        let item_tree = loc.id.item_tree(db);

        let mut flags = TypeAliasFlags::empty();
        let attrs = item_tree.attrs(
            db,
            loc.container.module(db).krate(),
            ModItem::from(loc.id.value).into(),
        );
        if attrs.by_key(sym::rustc_has_incoherent_inherent_impls).exists() {
            flags.insert(TypeAliasFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPL);
        }
        if attrs.by_key(sym::rustc_allow_incoherent_impl).exists() {
            flags.insert(TypeAliasFlags::RUSTC_ALLOW_INCOHERENT_IMPL);
        }
        if matches!(loc.container, ItemContainerId::ExternBlockId(_)) {
            flags.insert(TypeAliasFlags::IS_EXTERN);
        }
        let source = loc.source(db);
        let (store, source_map, generic_params, bounds, ty) =
            lower_type_alias(db, loc.container.module(db), source, id);

        (
            Arc::new(TypeAliasSignature {
                store: Arc::new(store),
                generic_params,
                flags,
                bounds,
                name: item_tree[loc.id.value].name.clone(),
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
}

pub type LocalFieldId = Idx<FieldData>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantFields {
    fields: Arena<FieldData>,
    pub store: Arc<ExpressionStore>,
    pub shape: FieldsShape,
}
impl VariantFields {
    #[inline]
    pub(crate) fn query(
        db: &dyn DefDatabase,
        id: VariantId,
    ) -> (Arc<Self>, Arc<ExpressionStoreSourceMap>) {
        let (shape, (fields, store, source_map)) = match id {
            VariantId::EnumVariantId(id) => {
                let loc = id.lookup(db);
                let item_tree = loc.id.item_tree(db);
                let parent = loc.parent.lookup(db);
                let variant = &item_tree[loc.id.value];
                (
                    variant.shape,
                    lower_fields(
                        db,
                        parent.container,
                        &item_tree,
                        FieldParent::EnumVariant(loc.id.value),
                        loc.source(db).map(|src| {
                            variant.fields.iter().zip(
                                src.field_list()
                                    .map(|it| {
                                        match it {
                                            ast::FieldList::RecordFieldList(record_field_list) => {
                                                Either::Left(record_field_list.fields().map(|it| {
                                                    (SyntaxNodePtr::new(it.syntax()), it.ty())
                                                }))
                                            }
                                            ast::FieldList::TupleFieldList(field_list) => {
                                                Either::Right(field_list.fields().map(|it| {
                                                    (SyntaxNodePtr::new(it.syntax()), it.ty())
                                                }))
                                            }
                                        }
                                        .into_iter()
                                    })
                                    .into_iter()
                                    .flatten(),
                            )
                        }),
                        Some(item_tree[parent.id.value].visibility),
                    ),
                )
            }
            VariantId::StructId(id) => {
                let loc = id.lookup(db);
                let item_tree = loc.id.item_tree(db);
                let strukt = &item_tree[loc.id.value];
                (
                    strukt.shape,
                    lower_fields(
                        db,
                        loc.container,
                        &item_tree,
                        FieldParent::Struct(loc.id.value),
                        loc.source(db).map(|src| {
                            strukt.fields.iter().zip(
                                src.field_list()
                                    .map(|it| {
                                        match it {
                                            ast::FieldList::RecordFieldList(record_field_list) => {
                                                Either::Left(record_field_list.fields().map(|it| {
                                                    (SyntaxNodePtr::new(it.syntax()), it.ty())
                                                }))
                                            }
                                            ast::FieldList::TupleFieldList(field_list) => {
                                                Either::Right(field_list.fields().map(|it| {
                                                    (SyntaxNodePtr::new(it.syntax()), it.ty())
                                                }))
                                            }
                                        }
                                        .into_iter()
                                    })
                                    .into_iter()
                                    .flatten(),
                            )
                        }),
                        None,
                    ),
                )
            }
            VariantId::UnionId(id) => {
                let loc = id.lookup(db);
                let item_tree = loc.id.item_tree(db);
                let union = &item_tree[loc.id.value];
                (
                    FieldsShape::Record,
                    lower_fields(
                        db,
                        loc.container,
                        &item_tree,
                        FieldParent::Union(loc.id.value),
                        loc.source(db).map(|src| {
                            union.fields.iter().zip(
                                src.record_field_list()
                                    .map(|it| {
                                        it.fields()
                                            .map(|it| (SyntaxNodePtr::new(it.syntax()), it.ty()))
                                    })
                                    .into_iter()
                                    .flatten(),
                            )
                        }),
                        None,
                    ),
                )
            }
        };

        (Arc::new(VariantFields { fields, store: Arc::new(store), shape }), Arc::new(source_map))
    }

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

fn lower_fields<'a>(
    db: &dyn DefDatabase,
    module: ModuleId,
    item_tree: &ItemTree,
    parent: FieldParent,
    fields: InFile<impl Iterator<Item = (&'a Field, (SyntaxNodePtr, Option<ast::Type>))>>,
    override_visibility: Option<RawVisibilityId>,
) -> (Arena<FieldData>, ExpressionStore, ExpressionStoreSourceMap) {
    let mut arena = Arena::new();
    let cfg_options = module.krate.cfg_options(db);
    let mut col = ExprCollector::new(db, module, fields.file_id);
    for (idx, (field, (ptr, ty))) in fields.value.enumerate() {
        let attr_owner = AttrOwner::make_field_indexed(parent, idx);
        let attrs = item_tree.attrs(db, module.krate, attr_owner);
        if attrs.is_cfg_enabled(cfg_options) {
            arena.alloc(FieldData {
                name: field.name.clone(),
                type_ref: col
                    .lower_type_ref_opt(ty, &mut ExprCollector::impl_trait_error_allocator),
                visibility: item_tree[override_visibility.unwrap_or(field.visibility)].clone(),
                is_unsafe: field.is_unsafe,
            });
        } else {
            col.source_map.diagnostics.push(
                crate::expr_store::ExpressionStoreDiagnostics::InactiveCode {
                    node: InFile::new(fields.file_id, ptr),
                    cfg: attrs.cfg().unwrap(),
                    opts: cfg_options.clone(),
                },
            );
        }
    }
    let store = col.store.finish();
    (arena, store, col.source_map)
}

#[derive(Debug, PartialEq, Eq)]
pub struct InactiveEnumVariantCode {
    pub cfg: CfgExpr,
    pub opts: CfgOptions,
    pub ast_id: span::FileAstId<ast::Variant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumVariants {
    pub variants: Box<[(EnumVariantId, Name)]>,
}

impl EnumVariants {
    pub(crate) fn enum_variants_query(
        db: &dyn DefDatabase,
        e: EnumId,
    ) -> (Arc<EnumVariants>, Option<Arc<ThinVec<InactiveEnumVariantCode>>>) {
        let loc = e.lookup(db);
        let item_tree = loc.id.item_tree(db);

        let mut diagnostics = ThinVec::new();
        let cfg_options = loc.container.krate.cfg_options(db);
        let mut index = 0;
        let variants = FileItemTreeId::range_iter(item_tree[loc.id.value].variants.clone())
            .filter_map(|variant| {
                let attrs = item_tree.attrs(db, loc.container.krate, variant.into());
                if attrs.is_cfg_enabled(cfg_options) {
                    let enum_variant = EnumVariantLoc {
                        id: ItemTreeId::new(loc.id.tree_id(), variant),
                        parent: e,
                        index,
                    }
                    .intern(db);
                    index += 1;
                    Some((enum_variant, item_tree[variant].name.clone()))
                } else {
                    diagnostics.push(InactiveEnumVariantCode {
                        ast_id: item_tree[variant].ast_id,
                        cfg: attrs.cfg().unwrap(),
                        opts: cfg_options.clone(),
                    });
                    None
                }
            })
            .collect();

        (
            Arc::new(EnumVariants { variants }),
            diagnostics.is_empty().not().then(|| Arc::new(diagnostics)),
        )
    }

    pub fn variant(&self, name: &Name) -> Option<EnumVariantId> {
        self.variants.iter().find_map(|(v, n)| if n == name { Some(*v) } else { None })
    }

    // [Adopted from rustc](https://github.com/rust-lang/rust/blob/bd53aa3bf7a24a70d763182303bd75e5fc51a9af/compiler/rustc_middle/src/ty/adt.rs#L446-L448)
    pub fn is_payload_free(&self, db: &dyn DefDatabase) -> bool {
        self.variants.iter().all(|&(v, _)| {
            // The condition check order is slightly modified from rustc
            // to improve performance by early returning with relatively fast checks
            let variant = &db.variant_fields(v.into());
            if !variant.fields().is_empty() {
                return false;
            }
            // The outer if condition is whether this variant has const ctor or not
            if !matches!(variant.shape, FieldsShape::Unit) {
                let body = db.body(v.into());
                // A variant with explicit discriminant
                if body.exprs[body.body_expr] != crate::hir::Expr::Missing {
                    return false;
                }
            }
            true
        })
    }
}
