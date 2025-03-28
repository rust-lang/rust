//! Defines hir-level representation of structs, enums and unions

use base_db::Crate;
use bitflags::bitflags;
use cfg::CfgOptions;

use hir_expand::name::Name;
use intern::sym;
use la_arena::Arena;
use rustc_abi::{IntegerType, ReprOptions};
use triomphe::Arc;

use crate::{
    EnumId, EnumVariantId, LocalFieldId, LocalModuleId, Lookup, StructId, UnionId, VariantId,
    db::DefDatabase,
    hir::Expr,
    item_tree::{
        AttrOwner, Field, FieldParent, FieldsShape, ItemTree, ModItem, RawVisibilityId, TreeId,
    },
    lang_item::LangItem,
    nameres::diagnostics::{DefDiagnostic, DefDiagnostics},
    type_ref::{TypeRefId, TypesMap},
    visibility::RawVisibility,
};

/// Note that we use `StructData` for unions as well!
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructData {
    pub name: Name,
    pub repr: Option<ReprOptions>,
    pub visibility: RawVisibility,
    pub flags: StructFlags,
}

bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub struct StructFlags: u8 {
        const NO_FLAGS         = 0;
        /// Indicates whether the struct is `PhantomData`.
        const IS_PHANTOM_DATA  = 1 << 2;
        /// Indicates whether the struct has a `#[fundamental]` attribute.
        const IS_FUNDAMENTAL   = 1 << 3;
        // FIXME: should this be a flag?
        /// Indicates whether the struct has a `#[rustc_has_incoherent_inherent_impls]` attribute.
        const IS_RUSTC_HAS_INCOHERENT_INHERENT_IMPL      = 1 << 4;
        /// Indicates whether this struct is `Box`.
        const IS_BOX           = 1 << 5;
        /// Indicates whether this struct is `ManuallyDrop`.
        const IS_MANUALLY_DROP = 1 << 6;
        /// Indicates whether this struct is `UnsafeCell`.
        const IS_UNSAFE_CELL   = 1 << 7;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumData {
    pub name: Name,
    pub repr: Option<ReprOptions>,
    pub visibility: RawVisibility,
    pub rustc_has_incoherent_inherent_impls: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumVariants {
    pub variants: Box<[(EnumVariantId, Name)]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumVariantData {
    pub name: Name,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariantData {
    Record { fields: Arena<FieldData>, types_map: Arc<TypesMap> },
    Tuple { fields: Arena<FieldData>, types_map: Arc<TypesMap> },
    Unit,
}

impl VariantData {
    #[inline]
    pub(crate) fn variant_data_query(db: &dyn DefDatabase, id: VariantId) -> Arc<VariantData> {
        db.variant_data_with_diagnostics(id).0
    }

    pub(crate) fn variant_data_with_diagnostics_query(
        db: &dyn DefDatabase,
        id: VariantId,
    ) -> (Arc<VariantData>, DefDiagnostics) {
        let (shape, types_map, (fields, diagnostics)) = match id {
            VariantId::EnumVariantId(id) => {
                let loc = id.lookup(db);
                let item_tree = loc.id.item_tree(db);
                let parent = loc.parent.lookup(db);
                let krate = parent.container.krate;
                let variant = &item_tree[loc.id.value];
                (
                    variant.shape,
                    variant.types_map.clone(),
                    lower_fields(
                        db,
                        krate,
                        parent.container.local_id,
                        loc.id.tree_id(),
                        &item_tree,
                        krate.cfg_options(db),
                        FieldParent::EnumVariant(loc.id.value),
                        &variant.fields,
                        Some(item_tree[parent.id.value].visibility),
                    ),
                )
            }
            VariantId::StructId(id) => {
                let loc = id.lookup(db);
                let item_tree = loc.id.item_tree(db);
                let krate = loc.container.krate;
                let strukt = &item_tree[loc.id.value];
                (
                    strukt.shape,
                    strukt.types_map.clone(),
                    lower_fields(
                        db,
                        krate,
                        loc.container.local_id,
                        loc.id.tree_id(),
                        &item_tree,
                        krate.cfg_options(db),
                        FieldParent::Struct(loc.id.value),
                        &strukt.fields,
                        None,
                    ),
                )
            }
            VariantId::UnionId(id) => {
                let loc = id.lookup(db);
                let item_tree = loc.id.item_tree(db);
                let krate = loc.container.krate;
                let union = &item_tree[loc.id.value];
                (
                    FieldsShape::Record,
                    union.types_map.clone(),
                    lower_fields(
                        db,
                        krate,
                        loc.container.local_id,
                        loc.id.tree_id(),
                        &item_tree,
                        krate.cfg_options(db),
                        FieldParent::Union(loc.id.value),
                        &union.fields,
                        None,
                    ),
                )
            }
        };

        (
            Arc::new(match shape {
                FieldsShape::Record => VariantData::Record { fields, types_map },
                FieldsShape::Tuple => VariantData::Tuple { fields, types_map },
                FieldsShape::Unit => VariantData::Unit,
            }),
            DefDiagnostics::new(diagnostics),
        )
    }
}

/// A single field of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldData {
    pub name: Name,
    pub type_ref: TypeRefId,
    pub visibility: RawVisibility,
    pub is_unsafe: bool,
}

fn repr_from_value(
    db: &dyn DefDatabase,
    krate: Crate,
    item_tree: &ItemTree,
    of: AttrOwner,
) -> Option<ReprOptions> {
    item_tree.attrs(db, krate, of).repr()
}

impl StructData {
    #[inline]
    pub(crate) fn struct_data_query(db: &dyn DefDatabase, id: StructId) -> Arc<StructData> {
        let loc = id.lookup(db);
        let krate = loc.container.krate;
        let item_tree = loc.id.item_tree(db);
        let repr = repr_from_value(db, krate, &item_tree, ModItem::from(loc.id.value).into());
        let attrs = item_tree.attrs(db, krate, ModItem::from(loc.id.value).into());

        let mut flags = StructFlags::NO_FLAGS;
        if attrs.by_key(&sym::rustc_has_incoherent_inherent_impls).exists() {
            flags |= StructFlags::IS_RUSTC_HAS_INCOHERENT_INHERENT_IMPL;
        }
        if attrs.by_key(&sym::fundamental).exists() {
            flags |= StructFlags::IS_FUNDAMENTAL;
        }
        if let Some(lang) = attrs.lang_item() {
            match lang {
                LangItem::PhantomData => flags |= StructFlags::IS_PHANTOM_DATA,
                LangItem::OwnedBox => flags |= StructFlags::IS_BOX,
                LangItem::ManuallyDrop => flags |= StructFlags::IS_MANUALLY_DROP,
                LangItem::UnsafeCell => flags |= StructFlags::IS_UNSAFE_CELL,
                _ => (),
            }
        }

        let strukt = &item_tree[loc.id.value];
        Arc::new(StructData {
            name: strukt.name.clone(),
            repr,
            visibility: item_tree[strukt.visibility].clone(),
            flags,
        })
    }

    #[inline]
    pub(crate) fn union_data_query(db: &dyn DefDatabase, id: UnionId) -> Arc<StructData> {
        let loc = id.lookup(db);
        let krate = loc.container.krate;
        let item_tree = loc.id.item_tree(db);
        let repr = repr_from_value(db, krate, &item_tree, ModItem::from(loc.id.value).into());
        let attrs = item_tree.attrs(db, krate, ModItem::from(loc.id.value).into());
        let mut flags = StructFlags::NO_FLAGS;

        if attrs.by_key(&sym::rustc_has_incoherent_inherent_impls).exists() {
            flags |= StructFlags::IS_RUSTC_HAS_INCOHERENT_INHERENT_IMPL;
        }
        if attrs.by_key(&sym::fundamental).exists() {
            flags |= StructFlags::IS_FUNDAMENTAL;
        }

        let union = &item_tree[loc.id.value];

        Arc::new(StructData {
            name: union.name.clone(),
            repr,
            visibility: item_tree[union.visibility].clone(),
            flags,
        })
    }
}

impl EnumVariants {
    pub(crate) fn enum_variants_query(db: &dyn DefDatabase, e: EnumId) -> Arc<EnumVariants> {
        let loc = e.lookup(db);
        let item_tree = loc.id.item_tree(db);

        Arc::new(EnumVariants {
            variants: loc.container.def_map(db).enum_definitions[&e]
                .iter()
                .map(|&id| (id, item_tree[id.lookup(db).id.value].name.clone()))
                .collect(),
        })
    }

    pub fn variant(&self, name: &Name) -> Option<EnumVariantId> {
        let &(id, _) = self.variants.iter().find(|(_id, n)| n == name)?;
        Some(id)
    }

    // [Adopted from rustc](https://github.com/rust-lang/rust/blob/bd53aa3bf7a24a70d763182303bd75e5fc51a9af/compiler/rustc_middle/src/ty/adt.rs#L446-L448)
    pub fn is_payload_free(&self, db: &dyn DefDatabase) -> bool {
        self.variants.iter().all(|&(v, _)| {
            // The condition check order is slightly modified from rustc
            // to improve performance by early returning with relatively fast checks
            let variant = &db.variant_data(v.into());
            if !variant.fields().is_empty() {
                return false;
            }
            // The outer if condition is whether this variant has const ctor or not
            if !matches!(variant.kind(), StructKind::Unit) {
                let body = db.body(v.into());
                // A variant with explicit discriminant
                if body.exprs[body.body_expr] != Expr::Missing {
                    return false;
                }
            }
            true
        })
    }
}

impl EnumData {
    pub(crate) fn enum_data_query(db: &dyn DefDatabase, e: EnumId) -> Arc<EnumData> {
        let loc = e.lookup(db);
        let krate = loc.container.krate;
        let item_tree = loc.id.item_tree(db);
        let repr = repr_from_value(db, krate, &item_tree, ModItem::from(loc.id.value).into());
        let attrs = item_tree.attrs(db, loc.container.krate, ModItem::from(loc.id.value).into());

        let rustc_has_incoherent_inherent_impls =
            attrs.by_key(&sym::rustc_has_incoherent_inherent_impls).exists();

        let enum_ = &item_tree[loc.id.value];

        Arc::new(EnumData {
            name: enum_.name.clone(),
            repr,
            visibility: item_tree[enum_.visibility].clone(),
            rustc_has_incoherent_inherent_impls,
        })
    }

    pub fn variant_body_type(&self) -> IntegerType {
        match self.repr {
            Some(ReprOptions { int: Some(builtin), .. }) => builtin,
            _ => IntegerType::Pointer(true),
        }
    }
}

impl EnumVariantData {
    #[inline]
    pub(crate) fn enum_variant_data_query(
        db: &dyn DefDatabase,
        e: EnumVariantId,
    ) -> Arc<EnumVariantData> {
        let loc = e.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let variant = &item_tree[loc.id.value];

        Arc::new(EnumVariantData { name: variant.name.clone() })
    }
}

impl VariantData {
    pub fn fields(&self) -> &Arena<FieldData> {
        const EMPTY: &Arena<FieldData> = &Arena::new();
        match self {
            VariantData::Record { fields, .. } | VariantData::Tuple { fields, .. } => fields,
            _ => EMPTY,
        }
    }

    pub fn types_map(&self) -> &TypesMap {
        match self {
            VariantData::Record { types_map, .. } | VariantData::Tuple { types_map, .. } => {
                types_map
            }
            VariantData::Unit => TypesMap::EMPTY,
        }
    }

    // FIXME: Linear lookup
    pub fn field(&self, name: &Name) -> Option<LocalFieldId> {
        self.fields().iter().find_map(|(id, data)| if &data.name == name { Some(id) } else { None })
    }

    pub fn kind(&self) -> StructKind {
        match self {
            VariantData::Record { .. } => StructKind::Record,
            VariantData::Tuple { .. } => StructKind::Tuple,
            VariantData::Unit => StructKind::Unit,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum StructKind {
    Tuple,
    Record,
    Unit,
}

fn lower_fields(
    db: &dyn DefDatabase,
    krate: Crate,
    container: LocalModuleId,
    tree_id: TreeId,
    item_tree: &ItemTree,
    cfg_options: &CfgOptions,
    parent: FieldParent,
    fields: &[Field],
    override_visibility: Option<RawVisibilityId>,
) -> (Arena<FieldData>, Vec<DefDiagnostic>) {
    let mut diagnostics = Vec::new();
    let mut arena = Arena::new();
    for (idx, field) in fields.iter().enumerate() {
        let attr_owner = AttrOwner::make_field_indexed(parent, idx);
        let attrs = item_tree.attrs(db, krate, attr_owner);
        if attrs.is_cfg_enabled(cfg_options) {
            arena.alloc(lower_field(item_tree, field, override_visibility));
        } else {
            diagnostics.push(DefDiagnostic::unconfigured_code(
                container,
                tree_id,
                attr_owner,
                attrs.cfg().unwrap(),
                cfg_options.clone(),
            ))
        }
    }
    (arena, diagnostics)
}

fn lower_field(
    item_tree: &ItemTree,
    field: &Field,
    override_visibility: Option<RawVisibilityId>,
) -> FieldData {
    FieldData {
        name: field.name.clone(),
        type_ref: field.type_ref,
        visibility: item_tree[override_visibility.unwrap_or(field.visibility)].clone(),
        is_unsafe: field.is_unsafe,
    }
}
