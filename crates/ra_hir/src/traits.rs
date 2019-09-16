//! HIR for trait definitions.

use rustc_hash::FxHashMap;
use std::sync::Arc;

use ra_syntax::ast::{self, NameOwner};

use crate::{
    db::{AstDatabase, DefDatabase},
    ids::LocationCtx,
    name::AsName,
    AssocItem, Const, Function, HasSource, Module, Name, Trait, TypeAlias,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitData {
    name: Option<Name>,
    items: Vec<AssocItem>,
    auto: bool,
}

impl TraitData {
    pub(crate) fn trait_data_query(
        db: &(impl DefDatabase + AstDatabase),
        tr: Trait,
    ) -> Arc<TraitData> {
        let src = tr.source(db);
        let name = src.ast.name().map(|n| n.as_name());
        let module = tr.module(db);
        let ctx = LocationCtx::new(db, module, src.file_id);
        let auto = src.ast.is_auto();
        let items = if let Some(item_list) = src.ast.item_list() {
            item_list
                .impl_items()
                .map(|item_node| match item_node {
                    ast::ImplItem::FnDef(it) => Function { id: ctx.to_def(&it) }.into(),
                    ast::ImplItem::ConstDef(it) => Const { id: ctx.to_def(&it) }.into(),
                    ast::ImplItem::TypeAliasDef(it) => TypeAlias { id: ctx.to_def(&it) }.into(),
                })
                .collect()
        } else {
            Vec::new()
        };
        Arc::new(TraitData { name, items, auto })
    }

    pub(crate) fn name(&self) -> &Option<Name> {
        &self.name
    }

    pub(crate) fn items(&self) -> &[AssocItem] {
        &self.items
    }

    pub(crate) fn is_auto(&self) -> bool {
        self.auto
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitItemsIndex {
    traits_by_def: FxHashMap<AssocItem, Trait>,
}

impl TraitItemsIndex {
    pub(crate) fn trait_items_index(db: &impl DefDatabase, module: Module) -> TraitItemsIndex {
        let mut index = TraitItemsIndex { traits_by_def: FxHashMap::default() };
        for decl in module.declarations(db) {
            if let crate::ModuleDef::Trait(tr) = decl {
                for item in tr.trait_data(db).items() {
                    index.traits_by_def.insert(*item, tr);
                }
            }
        }
        index
    }

    pub(crate) fn get_parent_trait(&self, item: AssocItem) -> Option<Trait> {
        self.traits_by_def.get(&item).cloned()
    }
}
