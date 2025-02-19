//! Defines hir-level representation of structs, enums and unions

use base_db::CrateId;
use bitflags::bitflags;
use cfg::CfgOptions;
use either::Either;

use hir_expand::name::Name;
use intern::sym;
use la_arena::Arena;
use rustc_abi::{Align, Integer, IntegerType, ReprFlags, ReprOptions};
use rustc_hashes::Hash64;
use triomphe::Arc;
use tt::iter::TtElement;

use crate::{
    builtin_type::{BuiltinInt, BuiltinUint},
    db::DefDatabase,
    hir::Expr,
    item_tree::{
        AttrOwner, Field, FieldParent, FieldsShape, ItemTree, ModItem, RawVisibilityId, TreeId,
    },
    lang_item::LangItem,
    nameres::diagnostics::{DefDiagnostic, DefDiagnostics},
    tt::{Delimiter, DelimiterKind, Leaf, TopSubtree},
    type_ref::{TypeRefId, TypesMap},
    visibility::RawVisibility,
    EnumId, EnumVariantId, LocalFieldId, LocalModuleId, Lookup, StructId, UnionId, VariantId,
};

/// Note that we use `StructData` for unions as well!
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructData {
    pub name: Name,
    pub variant_data: Arc<VariantData>,
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
    pub variants: Box<[(EnumVariantId, Name)]>,
    pub repr: Option<ReprOptions>,
    pub visibility: RawVisibility,
    pub rustc_has_incoherent_inherent_impls: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumVariantData {
    pub name: Name,
    pub variant_data: Arc<VariantData>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariantData {
    Record { fields: Arena<FieldData>, types_map: Arc<TypesMap> },
    Tuple { fields: Arena<FieldData>, types_map: Arc<TypesMap> },
    Unit,
}

/// A single field of an enum variant or struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldData {
    pub name: Name,
    pub type_ref: TypeRefId,
    pub visibility: RawVisibility,
}

fn repr_from_value(
    db: &dyn DefDatabase,
    krate: CrateId,
    item_tree: &ItemTree,
    of: AttrOwner,
) -> Option<ReprOptions> {
    item_tree.attrs(db, krate, of).by_key(&sym::repr).tt_values().find_map(parse_repr_tt)
}

fn parse_repr_tt(tt: &TopSubtree) -> Option<ReprOptions> {
    match tt.top_subtree().delimiter {
        Delimiter { kind: DelimiterKind::Parenthesis, .. } => {}
        _ => return None,
    }

    let mut flags = ReprFlags::empty();
    let mut int = None;
    let mut max_align: Option<Align> = None;
    let mut min_pack: Option<Align> = None;

    let mut tts = tt.iter();
    while let Some(tt) = tts.next() {
        if let TtElement::Leaf(Leaf::Ident(ident)) = tt {
            flags.insert(match &ident.sym {
                s if *s == sym::packed => {
                    let pack = if let Some(TtElement::Subtree(_, mut tt_iter)) = tts.peek() {
                        tts.next();
                        if let Some(TtElement::Leaf(Leaf::Literal(lit))) = tt_iter.next() {
                            lit.symbol.as_str().parse().unwrap_or_default()
                        } else {
                            0
                        }
                    } else {
                        0
                    };
                    let pack = Align::from_bytes(pack).unwrap_or(Align::ONE);
                    min_pack =
                        Some(if let Some(min_pack) = min_pack { min_pack.min(pack) } else { pack });
                    ReprFlags::empty()
                }
                s if *s == sym::align => {
                    if let Some(TtElement::Subtree(_, mut tt_iter)) = tts.peek() {
                        tts.next();
                        if let Some(TtElement::Leaf(Leaf::Literal(lit))) = tt_iter.next() {
                            if let Ok(align) = lit.symbol.as_str().parse() {
                                let align = Align::from_bytes(align).ok();
                                max_align = max_align.max(align);
                            }
                        }
                    }
                    ReprFlags::empty()
                }
                s if *s == sym::C => ReprFlags::IS_C,
                s if *s == sym::transparent => ReprFlags::IS_TRANSPARENT,
                s if *s == sym::simd => ReprFlags::IS_SIMD,
                repr => {
                    if let Some(builtin) = BuiltinInt::from_suffix_sym(repr)
                        .map(Either::Left)
                        .or_else(|| BuiltinUint::from_suffix_sym(repr).map(Either::Right))
                    {
                        int = Some(match builtin {
                            Either::Left(bi) => match bi {
                                BuiltinInt::Isize => IntegerType::Pointer(true),
                                BuiltinInt::I8 => IntegerType::Fixed(Integer::I8, true),
                                BuiltinInt::I16 => IntegerType::Fixed(Integer::I16, true),
                                BuiltinInt::I32 => IntegerType::Fixed(Integer::I32, true),
                                BuiltinInt::I64 => IntegerType::Fixed(Integer::I64, true),
                                BuiltinInt::I128 => IntegerType::Fixed(Integer::I128, true),
                            },
                            Either::Right(bu) => match bu {
                                BuiltinUint::Usize => IntegerType::Pointer(false),
                                BuiltinUint::U8 => IntegerType::Fixed(Integer::I8, false),
                                BuiltinUint::U16 => IntegerType::Fixed(Integer::I16, false),
                                BuiltinUint::U32 => IntegerType::Fixed(Integer::I32, false),
                                BuiltinUint::U64 => IntegerType::Fixed(Integer::I64, false),
                                BuiltinUint::U128 => IntegerType::Fixed(Integer::I128, false),
                            },
                        });
                    }
                    ReprFlags::empty()
                }
            })
        }
    }

    Some(ReprOptions { int, align: max_align, pack: min_pack, flags, field_shuffle_seed: Hash64::ZERO })
}

impl StructData {
    #[inline]
    pub(crate) fn struct_data_query(db: &dyn DefDatabase, id: StructId) -> Arc<StructData> {
        db.struct_data_with_diagnostics(id).0
    }

    pub(crate) fn struct_data_with_diagnostics_query(
        db: &dyn DefDatabase,
        id: StructId,
    ) -> (Arc<StructData>, DefDiagnostics) {
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
        let (fields, diagnostics) = lower_fields(
            db,
            krate,
            loc.container.local_id,
            loc.id.tree_id(),
            &item_tree,
            &db.crate_graph()[krate].cfg_options,
            FieldParent::Struct(loc.id.value),
            &strukt.fields,
            None,
        );
        let types_map = strukt.types_map.clone();
        (
            Arc::new(StructData {
                name: strukt.name.clone(),
                variant_data: Arc::new(match strukt.shape {
                    FieldsShape::Record => VariantData::Record { fields, types_map },
                    FieldsShape::Tuple => VariantData::Tuple { fields, types_map },
                    FieldsShape::Unit => VariantData::Unit,
                }),
                repr,
                visibility: item_tree[strukt.visibility].clone(),
                flags,
            }),
            DefDiagnostics::new(diagnostics),
        )
    }

    #[inline]
    pub(crate) fn union_data_query(db: &dyn DefDatabase, id: UnionId) -> Arc<StructData> {
        db.union_data_with_diagnostics(id).0
    }

    pub(crate) fn union_data_with_diagnostics_query(
        db: &dyn DefDatabase,
        id: UnionId,
    ) -> (Arc<StructData>, DefDiagnostics) {
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
        let (fields, diagnostics) = lower_fields(
            db,
            krate,
            loc.container.local_id,
            loc.id.tree_id(),
            &item_tree,
            &db.crate_graph()[krate].cfg_options,
            FieldParent::Union(loc.id.value),
            &union.fields,
            None,
        );
        let types_map = union.types_map.clone();
        (
            Arc::new(StructData {
                name: union.name.clone(),
                variant_data: Arc::new(VariantData::Record { fields, types_map }),
                repr,
                visibility: item_tree[union.visibility].clone(),
                flags,
            }),
            DefDiagnostics::new(diagnostics),
        )
    }
}

impl EnumData {
    pub(crate) fn enum_data_query(db: &dyn DefDatabase, e: EnumId) -> Arc<EnumData> {
        let loc = e.lookup(db);
        let krate = loc.container.krate;
        let item_tree = loc.id.item_tree(db);
        let repr = repr_from_value(db, krate, &item_tree, ModItem::from(loc.id.value).into());
        let rustc_has_incoherent_inherent_impls = item_tree
            .attrs(db, loc.container.krate, ModItem::from(loc.id.value).into())
            .by_key(&sym::rustc_has_incoherent_inherent_impls)
            .exists();

        let enum_ = &item_tree[loc.id.value];

        Arc::new(EnumData {
            name: enum_.name.clone(),
            variants: loc.container.def_map(db).enum_definitions[&e]
                .iter()
                .map(|&id| (id, item_tree[id.lookup(db).id.value].name.clone()))
                .collect(),
            repr,
            visibility: item_tree[enum_.visibility].clone(),
            rustc_has_incoherent_inherent_impls,
        })
    }

    pub fn variant(&self, name: &Name) -> Option<EnumVariantId> {
        let &(id, _) = self.variants.iter().find(|(_id, n)| n == name)?;
        Some(id)
    }

    pub fn variant_body_type(&self) -> IntegerType {
        match self.repr {
            Some(ReprOptions { int: Some(builtin), .. }) => builtin,
            _ => IntegerType::Pointer(true),
        }
    }

    // [Adopted from rustc](https://github.com/rust-lang/rust/blob/bd53aa3bf7a24a70d763182303bd75e5fc51a9af/compiler/rustc_middle/src/ty/adt.rs#L446-L448)
    pub fn is_payload_free(&self, db: &dyn DefDatabase) -> bool {
        self.variants.iter().all(|(v, _)| {
            // The condition check order is slightly modified from rustc
            // to improve performance by early returning with relatively fast checks
            let variant = &db.enum_variant_data(*v).variant_data;
            if !variant.fields().is_empty() {
                return false;
            }
            // The outer if condition is whether this variant has const ctor or not
            if !matches!(variant.kind(), StructKind::Unit) {
                let body = db.body((*v).into());
                // A variant with explicit discriminant
                if body.exprs[body.body_expr] != Expr::Missing {
                    return false;
                }
            }
            true
        })
    }
}

impl EnumVariantData {
    #[inline]
    pub(crate) fn enum_variant_data_query(
        db: &dyn DefDatabase,
        e: EnumVariantId,
    ) -> Arc<EnumVariantData> {
        db.enum_variant_data_with_diagnostics(e).0
    }

    pub(crate) fn enum_variant_data_with_diagnostics_query(
        db: &dyn DefDatabase,
        e: EnumVariantId,
    ) -> (Arc<EnumVariantData>, DefDiagnostics) {
        let loc = e.lookup(db);
        let container = loc.parent.lookup(db).container;
        let krate = container.krate;
        let item_tree = loc.id.item_tree(db);
        let variant = &item_tree[loc.id.value];

        let (fields, diagnostics) = lower_fields(
            db,
            krate,
            container.local_id,
            loc.id.tree_id(),
            &item_tree,
            &db.crate_graph()[krate].cfg_options,
            FieldParent::Variant(loc.id.value),
            &variant.fields,
            Some(item_tree[loc.parent.lookup(db).id.value].visibility),
        );
        let types_map = variant.types_map.clone();

        (
            Arc::new(EnumVariantData {
                name: variant.name.clone(),
                variant_data: Arc::new(match variant.shape {
                    FieldsShape::Record => VariantData::Record { fields, types_map },
                    FieldsShape::Tuple => VariantData::Tuple { fields, types_map },
                    FieldsShape::Unit => VariantData::Unit,
                }),
            }),
            DefDiagnostics::new(diagnostics),
        )
    }
}

impl VariantData {
    pub fn fields(&self) -> &Arena<FieldData> {
        const EMPTY: &Arena<FieldData> = &Arena::new();
        match &self {
            VariantData::Record { fields, .. } | VariantData::Tuple { fields, .. } => fields,
            _ => EMPTY,
        }
    }

    pub fn types_map(&self) -> &TypesMap {
        match &self {
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

    #[allow(clippy::self_named_constructors)]
    pub(crate) fn variant_data(db: &dyn DefDatabase, id: VariantId) -> Arc<VariantData> {
        match id {
            VariantId::StructId(it) => db.struct_data(it).variant_data.clone(),
            VariantId::EnumVariantId(it) => db.enum_variant_data(it).variant_data.clone(),
            VariantId::UnionId(it) => db.union_data(it).variant_data.clone(),
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
    krate: CrateId,
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
    }
}
