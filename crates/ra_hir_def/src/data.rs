//! Contains basic data about various HIR declarations.

use std::sync::Arc;

use hir_expand::{
    hygiene::Hygiene,
    name::{name, AsName, Name},
    InFile,
};
use ra_prof::profile;
use ra_syntax::ast::{self, NameOwner, TypeAscriptionOwner, TypeBoundsOwner, VisibilityOwner};

use crate::{
    attr::Attrs,
    body::Expander,
    body::LowerCtx,
    db::DefDatabase,
    item_tree::{AssocItem, ItemTreeId, ModItem},
    path::{path, AssociatedTypeBinding, GenericArgs, Path},
    src::HasSource,
    type_ref::{Mutability, TypeBound, TypeRef},
    visibility::RawVisibility,
    AssocContainerId, AssocItemId, ConstId, ConstLoc, FunctionId, FunctionLoc, HasModule, ImplId,
    Intern, Lookup, ModuleId, StaticId, TraitId, TypeAliasId, TypeAliasLoc,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionData {
    pub name: Name,
    pub params: Vec<TypeRef>,
    pub ret_type: TypeRef,
    pub attrs: Attrs,
    /// True if the first param is `self`. This is relevant to decide whether this
    /// can be called as a method.
    pub has_self_param: bool,
    pub is_unsafe: bool,
    pub visibility: RawVisibility,
}

impl FunctionData {
    pub(crate) fn fn_data_query(db: &impl DefDatabase, func: FunctionId) -> Arc<FunctionData> {
        let loc = func.lookup(db);
        let src = loc.source(db);
        let ctx = LowerCtx::new(db, src.file_id);
        let name = src.value.name().map(|n| n.as_name()).unwrap_or_else(Name::missing);
        let mut params = Vec::new();
        let mut has_self_param = false;
        if let Some(param_list) = src.value.param_list() {
            if let Some(self_param) = param_list.self_param() {
                let self_type = if let Some(type_ref) = self_param.ascribed_type() {
                    TypeRef::from_ast(&ctx, type_ref)
                } else {
                    let self_type = TypeRef::Path(name![Self].into());
                    match self_param.kind() {
                        ast::SelfParamKind::Owned => self_type,
                        ast::SelfParamKind::Ref => {
                            TypeRef::Reference(Box::new(self_type), Mutability::Shared)
                        }
                        ast::SelfParamKind::MutRef => {
                            TypeRef::Reference(Box::new(self_type), Mutability::Mut)
                        }
                    }
                };
                params.push(self_type);
                has_self_param = true;
            }
            for param in param_list.params() {
                let type_ref = TypeRef::from_ast_opt(&ctx, param.ascribed_type());
                params.push(type_ref);
            }
        }
        let attrs = Attrs::new(&src.value, &Hygiene::new(db.upcast(), src.file_id));

        let ret_type = if let Some(type_ref) = src.value.ret_type().and_then(|rt| rt.type_ref()) {
            TypeRef::from_ast(&ctx, type_ref)
        } else {
            TypeRef::unit()
        };

        let ret_type = if src.value.async_token().is_some() {
            let future_impl = desugar_future_path(ret_type);
            let ty_bound = TypeBound::Path(future_impl);
            TypeRef::ImplTrait(vec![ty_bound])
        } else {
            ret_type
        };

        let is_unsafe = src.value.unsafe_token().is_some();

        let vis_default = RawVisibility::default_for_container(loc.container);
        let visibility =
            RawVisibility::from_ast_with_default(db, vis_default, src.map(|s| s.visibility()));

        let sig =
            FunctionData { name, params, ret_type, has_self_param, is_unsafe, visibility, attrs };
        Arc::new(sig)
    }
}

fn desugar_future_path(orig: TypeRef) -> Path {
    let path = path![core::future::Future];
    let mut generic_args: Vec<_> = std::iter::repeat(None).take(path.segments.len() - 1).collect();
    let mut last = GenericArgs::empty();
    last.bindings.push(AssociatedTypeBinding {
        name: name![Output],
        type_ref: Some(orig),
        bounds: Vec::new(),
    });
    generic_args.push(Some(Arc::new(last)));

    Path::from_known_path(path, generic_args)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAliasData {
    pub name: Name,
    pub type_ref: Option<TypeRef>,
    pub visibility: RawVisibility,
    pub bounds: Vec<TypeBound>,
}

impl TypeAliasData {
    pub(crate) fn type_alias_data_query(
        db: &dyn DefDatabase,
        typ: TypeAliasId,
    ) -> Arc<TypeAliasData> {
        let loc = typ.lookup(db);
        let node = loc.source(db);
        let name = node.value.name().map_or_else(Name::missing, |n| n.as_name());
        let lower_ctx = LowerCtx::new(db, node.file_id);
        let type_ref = node.value.type_ref().map(|it| TypeRef::from_ast(&lower_ctx, it));
        let vis_default = RawVisibility::default_for_container(loc.container);
        let visibility = RawVisibility::from_ast_with_default(
            db,
            vis_default,
            node.as_ref().map(|n| n.visibility()),
        );
        let bounds = if let Some(bound_list) = node.value.type_bound_list() {
            bound_list.bounds().map(|it| TypeBound::from_ast(&lower_ctx, it)).collect()
        } else {
            Vec::new()
        };
        Arc::new(TypeAliasData { name, type_ref, visibility, bounds })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitData {
    pub name: Name,
    pub items: Vec<(Name, AssocItemId)>,
    pub auto: bool,
}

impl TraitData {
    pub(crate) fn trait_data_query(db: &dyn DefDatabase, tr: TraitId) -> Arc<TraitData> {
        let tr_loc = tr.lookup(db);
        let item_tree = db.item_tree(tr_loc.id.file_id);
        let tr_def = &item_tree[tr_loc.id.value];
        let name = tr_def.name.clone();
        let auto = tr_def.auto;
        let module_id = tr_loc.container.module(db);
        let container = AssocContainerId::TraitId(tr);
        let mut expander = Expander::new(db, tr_loc.id.file_id, module_id);

        let items = collect_items(
            db,
            module_id,
            &mut expander,
            tr_def.items.iter().copied(),
            tr_loc.id.file_id,
            container,
            100,
        );

        Arc::new(TraitData { name, items, auto })
    }

    pub fn associated_types(&self) -> impl Iterator<Item = TypeAliasId> + '_ {
        self.items.iter().filter_map(|(_name, item)| match item {
            AssocItemId::TypeAliasId(t) => Some(*t),
            _ => None,
        })
    }

    pub fn associated_type_by_name(&self, name: &Name) -> Option<TypeAliasId> {
        self.items.iter().find_map(|(item_name, item)| match item {
            AssocItemId::TypeAliasId(t) if item_name == name => Some(*t),
            _ => None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplData {
    pub target_trait: Option<TypeRef>,
    pub target_type: TypeRef,
    pub items: Vec<AssocItemId>,
    pub is_negative: bool,
}

impl ImplData {
    pub(crate) fn impl_data_query(db: &dyn DefDatabase, id: ImplId) -> Arc<ImplData> {
        let _p = profile("impl_data_query");
        let impl_loc = id.lookup(db);

        let item_tree = db.item_tree(impl_loc.id.file_id);
        let impl_def = &item_tree[impl_loc.id.value];
        let target_trait = impl_def.target_trait.clone();
        let target_type = impl_def.target_type.clone();
        let is_negative = impl_def.is_negative;
        let module_id = impl_loc.container.module(db);
        let container = AssocContainerId::ImplId(id);
        let mut expander = Expander::new(db, impl_loc.id.file_id, module_id);

        let items = collect_items(
            db,
            module_id,
            &mut expander,
            impl_def.items.iter().copied(),
            impl_loc.id.file_id,
            container,
            100,
        );
        let items = items.into_iter().map(|(_, item)| item).collect();

        Arc::new(ImplData { target_trait, target_type, items, is_negative })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstData {
    /// const _: () = ();
    pub name: Option<Name>,
    pub type_ref: TypeRef,
    pub visibility: RawVisibility,
}

impl ConstData {
    pub(crate) fn const_data_query(db: &dyn DefDatabase, konst: ConstId) -> Arc<ConstData> {
        let loc = konst.lookup(db);
        let node = loc.source(db);
        let vis_default = RawVisibility::default_for_container(loc.container);
        Arc::new(ConstData::new(db, vis_default, node))
    }

    fn new<N: NameOwner + TypeAscriptionOwner + VisibilityOwner>(
        db: &dyn DefDatabase,
        vis_default: RawVisibility,
        node: InFile<N>,
    ) -> ConstData {
        let ctx = LowerCtx::new(db, node.file_id);
        let name = node.value.name().map(|n| n.as_name());
        let type_ref = TypeRef::from_ast_opt(&ctx, node.value.ascribed_type());
        let visibility =
            RawVisibility::from_ast_with_default(db, vis_default, node.map(|n| n.visibility()));
        ConstData { name, type_ref, visibility }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticData {
    pub name: Option<Name>,
    pub type_ref: TypeRef,
    pub visibility: RawVisibility,
    pub mutable: bool,
}

impl StaticData {
    pub(crate) fn static_data_query(db: &dyn DefDatabase, konst: StaticId) -> Arc<StaticData> {
        let node = konst.lookup(db).source(db);
        let ctx = LowerCtx::new(db, node.file_id);

        let name = node.value.name().map(|n| n.as_name());
        let type_ref = TypeRef::from_ast_opt(&ctx, node.value.ascribed_type());
        let mutable = node.value.mut_token().is_some();
        let visibility = RawVisibility::from_ast_with_default(
            db,
            RawVisibility::private(),
            node.map(|n| n.visibility()),
        );

        Arc::new(StaticData { name, type_ref, visibility, mutable })
    }
}

fn collect_items(
    db: &dyn DefDatabase,
    module: ModuleId,
    expander: &mut Expander,
    assoc_items: impl Iterator<Item = AssocItem>,
    file_id: crate::HirFileId,
    container: AssocContainerId,
    limit: usize,
) -> Vec<(Name, AssocItemId)> {
    if limit == 0 {
        return Vec::new();
    }

    let item_tree = db.item_tree(file_id);
    let cfg_options = db.crate_graph()[module.krate].cfg_options.clone();

    let mut items = Vec::new();
    for item in assoc_items {
        match item {
            AssocItem::Function(id) => {
                let item = &item_tree[id];
                if !item.attrs.is_cfg_enabled(&cfg_options) {
                    continue;
                }
                let def = FunctionLoc { container, id: ItemTreeId::new(file_id, id) }.intern(db);
                items.push((item.name.clone(), def.into()));
            }
            // FIXME: cfg?
            AssocItem::Const(id) => {
                let item = &item_tree[id];
                let name = if let Some(name) = item.name.clone() {
                    name
                } else {
                    continue;
                };
                let def = ConstLoc { container, id: ItemTreeId::new(file_id, id) }.intern(db);
                items.push((name, def.into()));
            }
            AssocItem::TypeAlias(id) => {
                let item = &item_tree[id];
                let def = TypeAliasLoc { container, id: ItemTreeId::new(file_id, id) }.intern(db);
                items.push((item.name.clone(), def.into()));
            }
            AssocItem::MacroCall(call) => {
                let call = &item_tree[call];
                let ast_id_map = db.ast_id_map(file_id);
                let root = db.parse_or_expand(file_id).unwrap();
                let call = ast_id_map.get(call.ast_id).to_node(&root);

                if let Some((mark, mac)) = expander.enter_expand(db, None, call) {
                    let src: InFile<ast::MacroItems> = expander.to_source(mac);
                    let item_tree = db.item_tree(src.file_id);
                    let iter =
                        item_tree.top_level_items().iter().filter_map(ModItem::as_assoc_item);
                    items.extend(collect_items(
                        db,
                        module,
                        expander,
                        iter,
                        src.file_id,
                        container,
                        limit - 1,
                    ));

                    expander.exit(db, mark);
                }
            }
        }
    }

    items
}
