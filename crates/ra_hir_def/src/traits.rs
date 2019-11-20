//! HIR for trait definitions.

use std::sync::Arc;

use hir_expand::{
    name::{AsName, Name},
    AstId,
};

use ra_syntax::ast::{self, NameOwner};

use crate::{
    db::DefDatabase2, AssocItemId, AstItemDef, ConstLoc, ContainerId, FunctionLoc, Intern, TraitId,
    TypeAliasLoc,
};

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
}
