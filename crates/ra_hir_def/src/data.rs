//! Contains basic data about various HIR declarations.

use std::sync::Arc;

use hir_expand::{
    name::{name, AsName, Name},
    AstId, InFile,
};
use ra_syntax::ast::{self, AstNode, ImplItem, ModuleItemOwner, NameOwner, TypeAscriptionOwner};

use crate::{
    db::DefDatabase,
    path::{path, GenericArgs, Path},
    src::HasSource,
    type_ref::{Mutability, TypeBound, TypeRef},
    AssocContainerId, AssocItemId, ConstId, ConstLoc, Expander, FunctionId, FunctionLoc, HasModule,
    ImplId, Intern, Lookup, ModuleId, StaticId, TraitId, TypeAliasId, TypeAliasLoc,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionData {
    pub name: Name,
    pub params: Vec<TypeRef>,
    pub ret_type: TypeRef,
    /// True if the first param is `self`. This is relevant to decide whether this
    /// can be called as a method.
    pub has_self_param: bool,
}

impl FunctionData {
    pub(crate) fn fn_data_query(db: &impl DefDatabase, func: FunctionId) -> Arc<FunctionData> {
        let src = func.lookup(db).source(db);
        let name = src.value.name().map(|n| n.as_name()).unwrap_or_else(Name::missing);
        let mut params = Vec::new();
        let mut has_self_param = false;
        if let Some(param_list) = src.value.param_list() {
            if let Some(self_param) = param_list.self_param() {
                let self_type = if let Some(type_ref) = self_param.ascribed_type() {
                    TypeRef::from_ast(type_ref)
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
                let type_ref = TypeRef::from_ast_opt(param.ascribed_type());
                params.push(type_ref);
            }
        }
        let ret_type = if let Some(type_ref) = src.value.ret_type().and_then(|rt| rt.type_ref()) {
            TypeRef::from_ast(type_ref)
        } else {
            TypeRef::unit()
        };

        let ret_type = if src.value.is_async() {
            let future_impl = desugar_future_path(ret_type);
            let ty_bound = TypeBound::Path(future_impl);
            TypeRef::ImplTrait(vec![ty_bound])
        } else {
            ret_type
        };

        let sig = FunctionData { name, params, ret_type, has_self_param };
        Arc::new(sig)
    }
}

fn desugar_future_path(orig: TypeRef) -> Path {
    let path = path![std::future::Future];
    let mut generic_args: Vec<_> = std::iter::repeat(None).take(path.segments.len() - 1).collect();
    let mut last = GenericArgs::empty();
    last.bindings.push((name![Output], orig));
    generic_args.push(Some(Arc::new(last)));

    Path::from_known_path(path, generic_args)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAliasData {
    pub name: Name,
    pub type_ref: Option<TypeRef>,
}

impl TypeAliasData {
    pub(crate) fn type_alias_data_query(
        db: &impl DefDatabase,
        typ: TypeAliasId,
    ) -> Arc<TypeAliasData> {
        let node = typ.lookup(db).source(db).value;
        let name = node.name().map_or_else(Name::missing, |n| n.as_name());
        let type_ref = node.type_ref().map(TypeRef::from_ast);
        Arc::new(TypeAliasData { name, type_ref })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitData {
    pub name: Name,
    pub items: Vec<(Name, AssocItemId)>,
    pub auto: bool,
}

impl TraitData {
    pub(crate) fn trait_data_query(db: &impl DefDatabase, tr: TraitId) -> Arc<TraitData> {
        let src = tr.lookup(db).source(db);
        let name = src.value.name().map_or_else(Name::missing, |n| n.as_name());
        let auto = src.value.is_auto();
        let ast_id_map = db.ast_id_map(src.file_id);

        let container = AssocContainerId::TraitId(tr);
        let items = if let Some(item_list) = src.value.item_list() {
            item_list
                .impl_items()
                .map(|item_node| match item_node {
                    ast::ImplItem::FnDef(it) => {
                        let name = it.name().map_or_else(Name::missing, |it| it.as_name());
                        let def = FunctionLoc {
                            container,
                            ast_id: AstId::new(src.file_id, ast_id_map.ast_id(&it)),
                        }
                        .intern(db)
                        .into();
                        (name, def)
                    }
                    ast::ImplItem::ConstDef(it) => {
                        let name = it.name().map_or_else(Name::missing, |it| it.as_name());
                        let def = ConstLoc {
                            container,
                            ast_id: AstId::new(src.file_id, ast_id_map.ast_id(&it)),
                        }
                        .intern(db)
                        .into();
                        (name, def)
                    }
                    ast::ImplItem::TypeAliasDef(it) => {
                        let name = it.name().map_or_else(Name::missing, |it| it.as_name());
                        let def = TypeAliasLoc {
                            container,
                            ast_id: AstId::new(src.file_id, ast_id_map.ast_id(&it)),
                        }
                        .intern(db)
                        .into();
                        (name, def)
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
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
    pub(crate) fn impl_data_query(db: &impl DefDatabase, id: ImplId) -> Arc<ImplData> {
        let impl_loc = id.lookup(db);
        let src = impl_loc.source(db);

        let target_trait = src.value.target_trait().map(TypeRef::from_ast);
        let target_type = TypeRef::from_ast_opt(src.value.target_type());
        let is_negative = src.value.is_negative();
        let module_id = impl_loc.container.module(db);

        let mut items = Vec::new();
        if let Some(item_list) = src.value.item_list() {
            items.extend(collect_impl_items(db, item_list.impl_items(), src.file_id, id));
            items.extend(collect_impl_items_in_macros(
                db,
                module_id,
                &src.with_value(item_list),
                id,
            ));
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
}

impl ConstData {
    pub(crate) fn const_data_query(db: &impl DefDatabase, konst: ConstId) -> Arc<ConstData> {
        let node = konst.lookup(db).source(db).value;
        Arc::new(ConstData::new(&node))
    }

    pub(crate) fn static_data_query(db: &impl DefDatabase, konst: StaticId) -> Arc<ConstData> {
        let node = konst.lookup(db).source(db).value;
        Arc::new(ConstData::new(&node))
    }

    fn new<N: NameOwner + TypeAscriptionOwner>(node: &N) -> ConstData {
        let name = node.name().map(|n| n.as_name());
        let type_ref = TypeRef::from_ast_opt(node.ascribed_type());
        ConstData { name, type_ref }
    }
}

fn collect_impl_items_in_macros(
    db: &impl DefDatabase,
    module_id: ModuleId,
    impl_block: &InFile<ast::ItemList>,
    id: ImplId,
) -> Vec<AssocItemId> {
    let mut expander = Expander::new(db, impl_block.file_id, module_id);
    let mut res = Vec::new();

    // We set a limit to protect against infinite recursion
    let limit = 100;

    for m in impl_block.value.syntax().children().filter_map(ast::MacroCall::cast) {
        res.extend(collect_impl_items_in_macro(db, &mut expander, m, id, limit))
    }

    res
}

fn collect_impl_items_in_macro(
    db: &impl DefDatabase,
    expander: &mut Expander,
    m: ast::MacroCall,
    id: ImplId,
    limit: usize,
) -> Vec<AssocItemId> {
    if limit == 0 {
        return Vec::new();
    }

    if let Some((mark, items)) = expander.enter_expand(db, m) {
        let items: InFile<ast::MacroItems> = expander.to_source(items);
        let mut res = collect_impl_items(
            db,
            items.value.items().filter_map(|it| ImplItem::cast(it.syntax().clone())),
            items.file_id,
            id,
        );
        // Recursive collect macros
        // Note that ast::ModuleItem do not include ast::MacroCall
        // We cannot use ModuleItemOwner::items here
        for it in items.value.syntax().children().filter_map(ast::MacroCall::cast) {
            res.extend(collect_impl_items_in_macro(db, expander, it, id, limit - 1))
        }
        expander.exit(db, mark);
        res
    } else {
        Vec::new()
    }
}

fn collect_impl_items(
    db: &impl DefDatabase,
    impl_items: impl Iterator<Item = ImplItem>,
    file_id: crate::HirFileId,
    id: ImplId,
) -> Vec<AssocItemId> {
    let items = db.ast_id_map(file_id);

    impl_items
        .map(|item_node| match item_node {
            ast::ImplItem::FnDef(it) => {
                let def = FunctionLoc {
                    container: AssocContainerId::ImplId(id),
                    ast_id: AstId::new(file_id, items.ast_id(&it)),
                }
                .intern(db);
                def.into()
            }
            ast::ImplItem::ConstDef(it) => {
                let def = ConstLoc {
                    container: AssocContainerId::ImplId(id),
                    ast_id: AstId::new(file_id, items.ast_id(&it)),
                }
                .intern(db);
                def.into()
            }
            ast::ImplItem::TypeAliasDef(it) => {
                let def = TypeAliasLoc {
                    container: AssocContainerId::ImplId(id),
                    ast_id: AstId::new(file_id, items.ast_id(&it)),
                }
                .intern(db);
                def.into()
            }
        })
        .collect()
}
