use std::sync::Arc;

use hir_expand::{
    name::{self, AsName, Name},
    AstId,
};
use ra_syntax::ast::{self, NameOwner, TypeAscriptionOwner};

use crate::{
    db::DefDatabase2,
    type_ref::{Mutability, TypeRef},
    AssocItemId, AstItemDef, ConstLoc, ContainerId, FunctionId, FunctionLoc, HasSource, ImplId,
    Intern, Lookup, TraitId, TypeAliasId, TypeAliasLoc,
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
    pub(crate) fn fn_data_query(db: &impl DefDatabase2, func: FunctionId) -> Arc<FunctionData> {
        let src = func.lookup(db).source(db);
        let name = src.value.name().map(|n| n.as_name()).unwrap_or_else(Name::missing);
        let mut params = Vec::new();
        let mut has_self_param = false;
        if let Some(param_list) = src.value.param_list() {
            if let Some(self_param) = param_list.self_param() {
                let self_type = if let Some(type_ref) = self_param.ascribed_type() {
                    TypeRef::from_ast(type_ref)
                } else {
                    let self_type = TypeRef::Path(name::SELF_TYPE.into());
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

        let sig = FunctionData { name, params, ret_type, has_self_param };
        Arc::new(sig)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAliasData {
    pub name: Name,
    pub type_ref: Option<TypeRef>,
}

impl TypeAliasData {
    pub(crate) fn type_alias_data_query(
        db: &impl DefDatabase2,
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
    pub name: Option<Name>,
    pub items: Vec<AssocItemId>,
    pub auto: bool,
}

impl TraitData {
    pub(crate) fn trait_data_query(db: &impl DefDatabase2, tr: TraitId) -> Arc<TraitData> {
        let src = tr.source(db);
        let name = src.value.name().map(|n| n.as_name());
        let auto = src.value.is_auto();
        let ast_id_map = db.ast_id_map(src.file_id);
        let items = if let Some(item_list) = src.value.item_list() {
            item_list
                .impl_items()
                .map(|item_node| match item_node {
                    ast::ImplItem::FnDef(it) => FunctionLoc {
                        container: ContainerId::TraitId(tr),
                        ast_id: AstId::new(src.file_id, ast_id_map.ast_id(&it)),
                    }
                    .intern(db)
                    .into(),
                    ast::ImplItem::ConstDef(it) => ConstLoc {
                        container: ContainerId::TraitId(tr),
                        ast_id: AstId::new(src.file_id, ast_id_map.ast_id(&it)),
                    }
                    .intern(db)
                    .into(),
                    ast::ImplItem::TypeAliasDef(it) => TypeAliasLoc {
                        container: ContainerId::TraitId(tr),
                        ast_id: AstId::new(src.file_id, ast_id_map.ast_id(&it)),
                    }
                    .intern(db)
                    .into(),
                })
                .collect()
        } else {
            Vec::new()
        };
        Arc::new(TraitData { name, items, auto })
    }

    pub fn associated_types(&self) -> impl Iterator<Item = TypeAliasId> + '_ {
        self.items.iter().filter_map(|item| match item {
            AssocItemId::TypeAliasId(t) => Some(*t),
            _ => None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplData {
    target_trait: Option<TypeRef>,
    target_type: TypeRef,
    items: Vec<AssocItemId>,
    negative: bool,
}

impl ImplData {
    pub(crate) fn impl_data_query(db: &impl DefDatabase2, id: ImplId) -> Arc<ImplData> {
        let src = id.source(db);
        let items = db.ast_id_map(src.file_id);

        let target_trait = src.value.target_trait().map(TypeRef::from_ast);
        let target_type = TypeRef::from_ast_opt(src.value.target_type());
        let negative = src.value.is_negative();

        let items = if let Some(item_list) = src.value.item_list() {
            item_list
                .impl_items()
                .map(|item_node| match item_node {
                    ast::ImplItem::FnDef(it) => {
                        let def = FunctionLoc {
                            container: ContainerId::ImplId(id),
                            ast_id: AstId::new(src.file_id, items.ast_id(&it)),
                        }
                        .intern(db);
                        def.into()
                    }
                    ast::ImplItem::ConstDef(it) => {
                        let def = ConstLoc {
                            container: ContainerId::ImplId(id),
                            ast_id: AstId::new(src.file_id, items.ast_id(&it)),
                        }
                        .intern(db);
                        def.into()
                    }
                    ast::ImplItem::TypeAliasDef(it) => {
                        let def = TypeAliasLoc {
                            container: ContainerId::ImplId(id),
                            ast_id: AstId::new(src.file_id, items.ast_id(&it)),
                        }
                        .intern(db);
                        def.into()
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        let res = ImplData { target_trait, target_type, items, negative };
        Arc::new(res)
    }

    pub fn target_trait(&self) -> Option<&TypeRef> {
        self.target_trait.as_ref()
    }

    pub fn target_type(&self) -> &TypeRef {
        &self.target_type
    }

    pub fn items(&self) -> &[AssocItemId] {
        &self.items
    }

    pub fn is_negative(&self) -> bool {
        self.negative
    }
}
