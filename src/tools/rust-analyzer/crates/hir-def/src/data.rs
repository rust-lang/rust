//! Contains basic data about various HIR declarations.

pub mod adt;

use base_db::Crate;
use hir_expand::name::Name;
use intern::{Symbol, sym};
use la_arena::{Idx, RawIdx};
use triomphe::Arc;

use crate::{
    ConstId, ExternCrateId, FunctionId, HasModule, ImplId, ItemContainerId, ItemLoc, Lookup,
    Macro2Id, MacroRulesId, ProcMacroId, StaticId, TraitAliasId, TraitId, TypeAliasId,
    db::DefDatabase,
    item_tree::{self, FnFlags, ModItem},
    nameres::proc_macro::{ProcMacroKind, parse_macro_name_and_helper_attrs},
    path::ImportAlias,
    type_ref::{TraitRef, TypeBound, TypeRefId, TypesMap},
    visibility::RawVisibility,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionData {
    pub name: Name,
    pub params: Box<[TypeRefId]>,
    pub ret_type: TypeRefId,
    pub visibility: RawVisibility,
    pub abi: Option<Symbol>,
    pub legacy_const_generics_indices: Option<Box<Box<[u32]>>>,
    pub rustc_allow_incoherent_impl: bool,
    pub types_map: Arc<TypesMap>,
    flags: FnFlags,
}

impl FunctionData {
    pub(crate) fn fn_data_query(db: &dyn DefDatabase, func: FunctionId) -> Arc<FunctionData> {
        let loc = func.lookup(db);
        let krate = loc.container.module(db).krate;
        let item_tree = loc.id.item_tree(db);
        let func = &item_tree[loc.id.value];
        let visibility = if let ItemContainerId::TraitId(trait_id) = loc.container {
            trait_vis(db, trait_id)
        } else {
            item_tree[func.visibility].clone()
        };

        let cfg_options = krate.cfg_options(db);
        let attr_owner = |idx| {
            item_tree::AttrOwner::Param(loc.id.value, Idx::from_raw(RawIdx::from(idx as u32)))
        };

        let mut flags = func.flags;
        if flags.contains(FnFlags::HAS_SELF_PARAM) {
            // If there's a self param in the syntax, but it is cfg'd out, remove the flag.
            let is_cfgd_out =
                !item_tree.attrs(db, krate, attr_owner(0usize)).is_cfg_enabled(cfg_options);
            if is_cfgd_out {
                cov_mark::hit!(cfgd_out_self_param);
                flags.remove(FnFlags::HAS_SELF_PARAM);
            }
        }
        if flags.contains(FnFlags::IS_VARARGS) {
            if let Some((_, param)) = func.params.iter().enumerate().rev().find(|&(idx, _)| {
                item_tree.attrs(db, krate, attr_owner(idx)).is_cfg_enabled(cfg_options)
            }) {
                if param.type_ref.is_some() {
                    flags.remove(FnFlags::IS_VARARGS);
                }
            } else {
                flags.remove(FnFlags::IS_VARARGS);
            }
        }

        let attrs = item_tree.attrs(db, krate, ModItem::from(loc.id.value).into());
        let rustc_allow_incoherent_impl = attrs.by_key(&sym::rustc_allow_incoherent_impl).exists();
        if flags.contains(FnFlags::HAS_UNSAFE_KW)
            && attrs.by_key(&sym::rustc_deprecated_safe_2024).exists()
        {
            flags.remove(FnFlags::HAS_UNSAFE_KW);
            flags.insert(FnFlags::DEPRECATED_SAFE_2024);
        }

        if attrs.by_key(&sym::target_feature).exists() {
            flags.insert(FnFlags::HAS_TARGET_FEATURE);
        }

        Arc::new(FunctionData {
            name: func.name.clone(),
            params: func
                .params
                .iter()
                .enumerate()
                .filter(|&(idx, _)| {
                    item_tree.attrs(db, krate, attr_owner(idx)).is_cfg_enabled(cfg_options)
                })
                .filter_map(|(_, param)| param.type_ref)
                .collect(),
            ret_type: func.ret_type,
            visibility,
            abi: func.abi.clone(),
            legacy_const_generics_indices: attrs.rustc_legacy_const_generics(),
            types_map: func.types_map.clone(),
            flags,
            rustc_allow_incoherent_impl,
        })
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
        self.flags.contains(FnFlags::HAS_DEFAULT_KW)
    }

    pub fn is_const(&self) -> bool {
        self.flags.contains(FnFlags::HAS_CONST_KW)
    }

    pub fn is_async(&self) -> bool {
        self.flags.contains(FnFlags::HAS_ASYNC_KW)
    }

    pub fn is_unsafe(&self) -> bool {
        self.flags.contains(FnFlags::HAS_UNSAFE_KW)
    }

    pub fn is_deprecated_safe_2024(&self) -> bool {
        self.flags.contains(FnFlags::DEPRECATED_SAFE_2024)
    }

    pub fn is_safe(&self) -> bool {
        self.flags.contains(FnFlags::HAS_SAFE_KW)
    }

    pub fn is_varargs(&self) -> bool {
        self.flags.contains(FnFlags::IS_VARARGS)
    }

    pub fn has_target_feature(&self) -> bool {
        self.flags.contains(FnFlags::HAS_TARGET_FEATURE)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAliasData {
    pub name: Name,
    pub type_ref: Option<TypeRefId>,
    pub visibility: RawVisibility,
    pub is_extern: bool,
    pub rustc_has_incoherent_inherent_impls: bool,
    pub rustc_allow_incoherent_impl: bool,
    /// Bounds restricting the type alias itself (eg. `type Ty: Bound;` in a trait or impl).
    pub bounds: Box<[TypeBound]>,
    pub types_map: Arc<TypesMap>,
}

impl TypeAliasData {
    pub(crate) fn type_alias_data_query(
        db: &dyn DefDatabase,
        typ: TypeAliasId,
    ) -> Arc<TypeAliasData> {
        let loc = typ.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let typ = &item_tree[loc.id.value];
        let visibility = if let ItemContainerId::TraitId(trait_id) = loc.container {
            trait_vis(db, trait_id)
        } else {
            item_tree[typ.visibility].clone()
        };

        let attrs = item_tree.attrs(
            db,
            loc.container.module(db).krate(),
            ModItem::from(loc.id.value).into(),
        );
        let rustc_has_incoherent_inherent_impls =
            attrs.by_key(&sym::rustc_has_incoherent_inherent_impls).exists();
        let rustc_allow_incoherent_impl = attrs.by_key(&sym::rustc_allow_incoherent_impl).exists();

        Arc::new(TypeAliasData {
            name: typ.name.clone(),
            type_ref: typ.type_ref,
            visibility,
            is_extern: matches!(loc.container, ItemContainerId::ExternBlockId(_)),
            rustc_has_incoherent_inherent_impls,
            rustc_allow_incoherent_impl,
            bounds: typ.bounds.clone(),
            types_map: typ.types_map.clone(),
        })
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
    pub struct TraitFlags: u8 {
        const IS_AUTO = 1 << 0;
        const IS_UNSAFE = 1 << 1;
        const IS_FUNDAMENTAL = 1 << 2;
        const RUSTC_HAS_INCOHERENT_INHERENT_IMPLS = 1 << 3;
        const SKIP_ARRAY_DURING_METHOD_DISPATCH = 1 << 4;
        const SKIP_BOXED_SLICE_DURING_METHOD_DISPATCH = 1 << 5;
        const RUSTC_PAREN_SUGAR = 1 << 6;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitData {
    pub name: Name,
    pub flags: TraitFlags,
    pub visibility: RawVisibility,
}

impl TraitData {
    #[inline]
    pub(crate) fn trait_data_query(db: &dyn DefDatabase, tr: TraitId) -> Arc<TraitData> {
        let ItemLoc { container: module_id, id: tree_id } = tr.lookup(db);
        let item_tree = tree_id.item_tree(db);
        let tr_def = &item_tree[tree_id.value];
        let name = tr_def.name.clone();
        let visibility = item_tree[tr_def.visibility].clone();
        let attrs = item_tree.attrs(db, module_id.krate(), ModItem::from(tree_id.value).into());

        let mut flags = TraitFlags::empty();

        if tr_def.is_auto {
            flags |= TraitFlags::IS_AUTO;
        }
        if tr_def.is_unsafe {
            flags |= TraitFlags::IS_UNSAFE;
        }
        if attrs.by_key(&sym::fundamental).exists() {
            flags |= TraitFlags::IS_FUNDAMENTAL;
        }
        if attrs.by_key(&sym::rustc_has_incoherent_inherent_impls).exists() {
            flags |= TraitFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS;
        }
        if attrs.by_key(&sym::rustc_paren_sugar).exists() {
            flags |= TraitFlags::RUSTC_PAREN_SUGAR;
        }

        let mut skip_array_during_method_dispatch =
            attrs.by_key(&sym::rustc_skip_array_during_method_dispatch).exists();
        let mut skip_boxed_slice_during_method_dispatch = false;
        for tt in attrs.by_key(&sym::rustc_skip_during_method_dispatch).tt_values() {
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

        Arc::new(TraitData { name, visibility, flags })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitAliasData {
    pub name: Name,
    pub visibility: RawVisibility,
}

impl TraitAliasData {
    pub(crate) fn trait_alias_query(db: &dyn DefDatabase, id: TraitAliasId) -> Arc<TraitAliasData> {
        let loc = id.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let alias = &item_tree[loc.id.value];
        let visibility = item_tree[alias.visibility].clone();

        Arc::new(TraitAliasData { name: alias.name.clone(), visibility })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ImplData {
    pub target_trait: Option<TraitRef>,
    pub self_ty: TypeRefId,
    pub is_negative: bool,
    pub is_unsafe: bool,
    pub types_map: Arc<TypesMap>,
}

impl ImplData {
    #[inline]
    pub(crate) fn impl_data_query(db: &dyn DefDatabase, id: ImplId) -> Arc<ImplData> {
        let _p = tracing::info_span!("impl_data_query").entered();
        let ItemLoc { id: tree_id, .. } = id.lookup(db);

        let item_tree = tree_id.item_tree(db);
        let impl_def = &item_tree[tree_id.value];
        let target_trait = impl_def.target_trait;
        let self_ty = impl_def.self_ty;
        let is_negative = impl_def.is_negative;
        let is_unsafe = impl_def.is_unsafe;

        Arc::new(ImplData {
            target_trait,
            self_ty,
            is_negative,
            is_unsafe,
            types_map: impl_def.types_map.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Macro2Data {
    pub name: Name,
    pub visibility: RawVisibility,
    // It's a bit wasteful as currently this is only for builtin `Default` derive macro, but macro2
    // are rarely used in practice so I think it's okay for now.
    /// Derive helpers, if this is a derive rustc_builtin_macro
    pub helpers: Option<Box<[Name]>>,
}

impl Macro2Data {
    pub(crate) fn macro2_data_query(db: &dyn DefDatabase, makro: Macro2Id) -> Arc<Macro2Data> {
        let loc = makro.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let makro = &item_tree[loc.id.value];

        let helpers = item_tree
            .attrs(db, loc.container.krate(), ModItem::from(loc.id.value).into())
            .by_key(&sym::rustc_builtin_macro)
            .tt_values()
            .next()
            .and_then(parse_macro_name_and_helper_attrs)
            .map(|(_, helpers)| helpers);

        Arc::new(Macro2Data {
            name: makro.name.clone(),
            visibility: item_tree[makro.visibility].clone(),
            helpers,
        })
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroRulesData {
    pub name: Name,
    pub macro_export: bool,
}

impl MacroRulesData {
    pub(crate) fn macro_rules_data_query(
        db: &dyn DefDatabase,
        makro: MacroRulesId,
    ) -> Arc<MacroRulesData> {
        let loc = makro.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let makro = &item_tree[loc.id.value];

        let macro_export = item_tree
            .attrs(db, loc.container.krate(), ModItem::from(loc.id.value).into())
            .by_key(&sym::macro_export)
            .exists();

        Arc::new(MacroRulesData { name: makro.name.clone(), macro_export })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcMacroData {
    pub name: Name,
    /// Derive helpers, if this is a derive
    pub helpers: Option<Box<[Name]>>,
}

impl ProcMacroData {
    pub(crate) fn proc_macro_data_query(
        db: &dyn DefDatabase,
        makro: ProcMacroId,
    ) -> Arc<ProcMacroData> {
        let loc = makro.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let makro = &item_tree[loc.id.value];

        let (name, helpers) = if let Some(def) = item_tree
            .attrs(db, loc.container.krate(), ModItem::from(loc.id.value).into())
            .parse_proc_macro_decl(&makro.name)
        {
            (
                def.name,
                match def.kind {
                    ProcMacroKind::Derive { helpers } => Some(helpers),
                    ProcMacroKind::Bang | ProcMacroKind::Attr => None,
                },
            )
        } else {
            // eeeh...
            stdx::never!("proc macro declaration is not a proc macro");
            (makro.name.clone(), None)
        };
        Arc::new(ProcMacroData { name, helpers })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternCrateDeclData {
    pub name: Name,
    pub alias: Option<ImportAlias>,
    pub visibility: RawVisibility,
    pub crate_id: Option<Crate>,
}

impl ExternCrateDeclData {
    pub(crate) fn extern_crate_decl_data_query(
        db: &dyn DefDatabase,
        extern_crate: ExternCrateId,
    ) -> Arc<ExternCrateDeclData> {
        let loc = extern_crate.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let extern_crate = &item_tree[loc.id.value];

        let name = extern_crate.name.clone();
        let krate = loc.container.krate();
        let crate_id = if name == sym::self_.clone() {
            Some(krate)
        } else {
            krate.data(db).dependencies.iter().find_map(|dep| {
                if dep.name.symbol() == name.symbol() { Some(dep.crate_id) } else { None }
            })
        };

        Arc::new(Self {
            name,
            visibility: item_tree[extern_crate.visibility].clone(),
            alias: extern_crate.alias.clone(),
            crate_id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstData {
    /// `None` for `const _: () = ();`
    pub name: Option<Name>,
    pub type_ref: TypeRefId,
    pub visibility: RawVisibility,
    pub rustc_allow_incoherent_impl: bool,
    pub has_body: bool,
    pub types_map: Arc<TypesMap>,
}

impl ConstData {
    pub(crate) fn const_data_query(db: &dyn DefDatabase, konst: ConstId) -> Arc<ConstData> {
        let loc = konst.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let konst = &item_tree[loc.id.value];
        let visibility = if let ItemContainerId::TraitId(trait_id) = loc.container {
            trait_vis(db, trait_id)
        } else {
            item_tree[konst.visibility].clone()
        };

        let rustc_allow_incoherent_impl = item_tree
            .attrs(db, loc.container.module(db).krate(), ModItem::from(loc.id.value).into())
            .by_key(&sym::rustc_allow_incoherent_impl)
            .exists();

        Arc::new(ConstData {
            name: konst.name.clone(),
            type_ref: konst.type_ref,
            visibility,
            rustc_allow_incoherent_impl,
            has_body: konst.has_body,
            types_map: konst.types_map.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticData {
    pub name: Name,
    pub type_ref: TypeRefId,
    pub visibility: RawVisibility,
    pub mutable: bool,
    pub is_extern: bool,
    pub has_safe_kw: bool,
    pub has_unsafe_kw: bool,
    pub types_map: Arc<TypesMap>,
}

impl StaticData {
    pub(crate) fn static_data_query(db: &dyn DefDatabase, konst: StaticId) -> Arc<StaticData> {
        let loc = konst.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let statik = &item_tree[loc.id.value];

        Arc::new(StaticData {
            name: statik.name.clone(),
            type_ref: statik.type_ref,
            visibility: item_tree[statik.visibility].clone(),
            mutable: statik.mutable,
            is_extern: matches!(loc.container, ItemContainerId::ExternBlockId(_)),
            has_safe_kw: statik.has_safe_kw,
            has_unsafe_kw: statik.has_unsafe_kw,
            types_map: statik.types_map.clone(),
        })
    }
}

fn trait_vis(db: &dyn DefDatabase, trait_id: TraitId) -> RawVisibility {
    let ItemLoc { id: tree_id, .. } = trait_id.lookup(db);
    let item_tree = tree_id.item_tree(db);
    let tr_def = &item_tree[tree_id.value];
    item_tree[tr_def.visibility].clone()
}
