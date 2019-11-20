//! HIR for trait definitions.

use std::sync::Arc;

use hir_expand::{
    name::{AsName, Name},
    AstId,
};

use ra_syntax::ast::{self, NameOwner};
use rustc_hash::FxHashMap;

use crate::{
    db::DefDatabase2, AssocItemId, AstItemDef, ConstId, ContainerId, FunctionLoc, Intern,
    LocationCtx, ModuleDefId, ModuleId, TraitId, TypeAliasLoc,
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
        let module = tr.module(db);
        let ctx = LocationCtx::new(db, module, src.file_id);
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
                    ast::ImplItem::ConstDef(it) => ConstId::from_ast(ctx, &it).into(),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitItemsIndex {
    traits_by_def: FxHashMap<AssocItemId, TraitId>,
}

impl TraitItemsIndex {
    pub fn trait_items_index(db: &impl DefDatabase2, module: ModuleId) -> TraitItemsIndex {
        let mut index = TraitItemsIndex { traits_by_def: FxHashMap::default() };
        let crate_def_map = db.crate_def_map(module.krate);
        for decl in crate_def_map[module.module_id].scope.declarations() {
            if let ModuleDefId::TraitId(tr) = decl {
                for item in db.trait_data(tr).items.iter() {
                    match item {
                        AssocItemId::FunctionId(_) => (),
                        AssocItemId::TypeAliasId(_) => (),
                        _ => {
                            let prev = index.traits_by_def.insert(*item, tr);
                            assert!(prev.is_none());
                        }
                    }
                }
            }
        }
        index
    }

    pub fn get_parent_trait(&self, item: AssocItemId) -> Option<TraitId> {
        self.traits_by_def.get(&item).cloned()
    }
}
