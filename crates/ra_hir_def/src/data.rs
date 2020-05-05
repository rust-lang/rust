//! Contains basic data about various HIR declarations.

use std::sync::Arc;

use hir_expand::{
    hygiene::Hygiene,
    name::{name, AsName, Name},
    AstId, InFile,
};
use ra_prof::profile;
use ra_syntax::ast::{
    self, AstNode, ImplItem, ModuleItemOwner, NameOwner, TypeAscriptionOwner, TypeBoundsOwner,
    VisibilityOwner,
};

use crate::{
    attr::Attrs,
    body::LowerCtx,
    db::DefDatabase,
    path::{path, AssociatedTypeBinding, GenericArgs, Path},
    src::HasSource,
    type_ref::{Mutability, TypeBound, TypeRef},
    visibility::RawVisibility,
    AssocContainerId, AssocItemId, ConstId, ConstLoc, Expander, FunctionId, FunctionLoc, HasModule,
    ImplId, Intern, Lookup, StaticId, TraitId, TypeAliasId, TypeAliasLoc,
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

        let vis_default = RawVisibility::default_for_container(loc.container);
        let visibility =
            RawVisibility::from_ast_with_default(db, vis_default, src.map(|s| s.visibility()));

        let sig = FunctionData { name, params, ret_type, has_self_param, visibility, attrs };
        Arc::new(sig)
    }
}

fn desugar_future_path(orig: TypeRef) -> Path {
    let path = path![std::future::Future];
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
        let src = tr_loc.source(db);
        let name = src.value.name().map_or_else(Name::missing, |n| n.as_name());
        let auto = src.value.auto_token().is_some();
        let module_id = tr_loc.container.module(db);

        let container = AssocContainerId::TraitId(tr);
        let mut items = Vec::new();

        if let Some(item_list) = src.value.item_list() {
            let mut expander = Expander::new(db, tr_loc.ast_id.file_id, module_id);
            items.extend(collect_items(
                db,
                &mut expander,
                item_list.impl_items(),
                src.file_id,
                container,
            ));
            items.extend(collect_items_in_macros(
                db,
                &mut expander,
                &src.with_value(item_list),
                container,
            ));
        }
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
        let src = impl_loc.source(db);
        let lower_ctx = LowerCtx::new(db, src.file_id);

        let target_trait = src.value.target_trait().map(|it| TypeRef::from_ast(&lower_ctx, it));
        let target_type = TypeRef::from_ast_opt(&lower_ctx, src.value.target_type());
        let is_negative = src.value.excl_token().is_some();
        let module_id = impl_loc.container.module(db);
        let container = AssocContainerId::ImplId(id);

        let mut items: Vec<AssocItemId> = Vec::new();

        if let Some(item_list) = src.value.item_list() {
            let mut expander = Expander::new(db, impl_loc.ast_id.file_id, module_id);
            items.extend(
                collect_items(db, &mut expander, item_list.impl_items(), src.file_id, container)
                    .into_iter()
                    .map(|(_, item)| item),
            );
            items.extend(
                collect_items_in_macros(db, &mut expander, &src.with_value(item_list), container)
                    .into_iter()
                    .map(|(_, item)| item),
            );
        }

        let res = ImplData { target_trait, target_type, items, is_negative };
        Arc::new(res)
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

    pub(crate) fn static_data_query(db: &dyn DefDatabase, konst: StaticId) -> Arc<ConstData> {
        let node = konst.lookup(db).source(db);
        Arc::new(ConstData::new(db, RawVisibility::private(), node))
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

fn collect_items_in_macros(
    db: &dyn DefDatabase,
    expander: &mut Expander,
    impl_def: &InFile<ast::ItemList>,
    container: AssocContainerId,
) -> Vec<(Name, AssocItemId)> {
    let mut res = Vec::new();

    // We set a limit to protect against infinite recursion
    let limit = 100;

    for m in impl_def.value.syntax().children().filter_map(ast::MacroCall::cast) {
        res.extend(collect_items_in_macro(db, expander, m, container, limit))
    }

    res
}

fn collect_items_in_macro(
    db: &dyn DefDatabase,
    expander: &mut Expander,
    m: ast::MacroCall,
    container: AssocContainerId,
    limit: usize,
) -> Vec<(Name, AssocItemId)> {
    if limit == 0 {
        return Vec::new();
    }

    if let Some((mark, items)) = expander.enter_expand(db, None, m) {
        let items: InFile<ast::MacroItems> = expander.to_source(items);
        let mut res = collect_items(
            db,
            expander,
            items.value.items().filter_map(|it| ImplItem::cast(it.syntax().clone())),
            items.file_id,
            container,
        );

        // Recursive collect macros
        // Note that ast::ModuleItem do not include ast::MacroCall
        // We cannot use ModuleItemOwner::items here
        for it in items.value.syntax().children().filter_map(ast::MacroCall::cast) {
            res.extend(collect_items_in_macro(db, expander, it, container, limit - 1))
        }
        expander.exit(db, mark);
        res
    } else {
        Vec::new()
    }
}

fn collect_items(
    db: &dyn DefDatabase,
    expander: &mut Expander,
    impl_items: impl Iterator<Item = ImplItem>,
    file_id: crate::HirFileId,
    container: AssocContainerId,
) -> Vec<(Name, AssocItemId)> {
    let items = db.ast_id_map(file_id);

    impl_items
        .filter_map(|item_node| match item_node {
            ast::ImplItem::FnDef(it) => {
                let name = it.name().map_or_else(Name::missing, |it| it.as_name());
                if !expander.is_cfg_enabled(&it) {
                    return None;
                }
                let def = FunctionLoc { container, ast_id: AstId::new(file_id, items.ast_id(&it)) }
                    .intern(db);
                Some((name, def.into()))
            }
            ast::ImplItem::ConstDef(it) => {
                let name = it.name().map_or_else(Name::missing, |it| it.as_name());
                let def = ConstLoc { container, ast_id: AstId::new(file_id, items.ast_id(&it)) }
                    .intern(db);
                Some((name, def.into()))
            }
            ast::ImplItem::TypeAliasDef(it) => {
                let name = it.name().map_or_else(Name::missing, |it| it.as_name());
                let def =
                    TypeAliasLoc { container, ast_id: AstId::new(file_id, items.ast_id(&it)) }
                        .intern(db);
                Some((name, def.into()))
            }
        })
        .collect()
}
