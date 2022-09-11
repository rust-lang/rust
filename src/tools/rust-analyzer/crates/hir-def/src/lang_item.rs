//! Collects lang items: items marked with `#[lang = "..."]` attribute.
//!
//! This attribute to tell the compiler about semi built-in std library
//! features, such as Fn family of traits.
use std::sync::Arc;

use rustc_hash::FxHashMap;
use syntax::SmolStr;

use crate::{
    db::DefDatabase, AdtId, AttrDefId, CrateId, EnumId, EnumVariantId, FunctionId, ImplId,
    ModuleDefId, StaticId, StructId, TraitId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LangItemTarget {
    EnumId(EnumId),
    FunctionId(FunctionId),
    ImplDefId(ImplId),
    StaticId(StaticId),
    StructId(StructId),
    TraitId(TraitId),
    EnumVariantId(EnumVariantId),
}

impl LangItemTarget {
    pub fn as_enum(self) -> Option<EnumId> {
        match self {
            LangItemTarget::EnumId(id) => Some(id),
            _ => None,
        }
    }

    pub fn as_function(self) -> Option<FunctionId> {
        match self {
            LangItemTarget::FunctionId(id) => Some(id),
            _ => None,
        }
    }

    pub fn as_impl_def(self) -> Option<ImplId> {
        match self {
            LangItemTarget::ImplDefId(id) => Some(id),
            _ => None,
        }
    }

    pub fn as_static(self) -> Option<StaticId> {
        match self {
            LangItemTarget::StaticId(id) => Some(id),
            _ => None,
        }
    }

    pub fn as_struct(self) -> Option<StructId> {
        match self {
            LangItemTarget::StructId(id) => Some(id),
            _ => None,
        }
    }

    pub fn as_trait(self) -> Option<TraitId> {
        match self {
            LangItemTarget::TraitId(id) => Some(id),
            _ => None,
        }
    }

    pub fn as_enum_variant(self) -> Option<EnumVariantId> {
        match self {
            LangItemTarget::EnumVariantId(id) => Some(id),
            _ => None,
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct LangItems {
    items: FxHashMap<SmolStr, LangItemTarget>,
}

impl LangItems {
    pub fn target(&self, item: &str) -> Option<LangItemTarget> {
        self.items.get(item).copied()
    }

    /// Salsa query. This will look for lang items in a specific crate.
    pub(crate) fn crate_lang_items_query(db: &dyn DefDatabase, krate: CrateId) -> Arc<LangItems> {
        let _p = profile::span("crate_lang_items_query");

        let mut lang_items = LangItems::default();

        let crate_def_map = db.crate_def_map(krate);

        for (_, module_data) in crate_def_map.modules() {
            for impl_def in module_data.scope.impls() {
                lang_items.collect_lang_item(db, impl_def, LangItemTarget::ImplDefId)
            }

            for def in module_data.scope.declarations() {
                match def {
                    ModuleDefId::TraitId(trait_) => {
                        lang_items.collect_lang_item(db, trait_, LangItemTarget::TraitId);
                        db.trait_data(trait_).items.iter().for_each(|&(_, assoc_id)| {
                            if let crate::AssocItemId::FunctionId(f) = assoc_id {
                                lang_items.collect_lang_item(db, f, LangItemTarget::FunctionId);
                            }
                        });
                    }
                    ModuleDefId::AdtId(AdtId::EnumId(e)) => {
                        lang_items.collect_lang_item(db, e, LangItemTarget::EnumId);
                        db.enum_data(e).variants.iter().for_each(|(local_id, _)| {
                            lang_items.collect_lang_item(
                                db,
                                EnumVariantId { parent: e, local_id },
                                LangItemTarget::EnumVariantId,
                            );
                        });
                    }
                    ModuleDefId::AdtId(AdtId::StructId(s)) => {
                        lang_items.collect_lang_item(db, s, LangItemTarget::StructId);
                    }
                    ModuleDefId::FunctionId(f) => {
                        lang_items.collect_lang_item(db, f, LangItemTarget::FunctionId);
                    }
                    ModuleDefId::StaticId(s) => {
                        lang_items.collect_lang_item(db, s, LangItemTarget::StaticId);
                    }
                    _ => {}
                }
            }
        }

        Arc::new(lang_items)
    }

    /// Salsa query. Look for a lang item, starting from the specified crate and recursively
    /// traversing its dependencies.
    pub(crate) fn lang_item_query(
        db: &dyn DefDatabase,
        start_crate: CrateId,
        item: SmolStr,
    ) -> Option<LangItemTarget> {
        let _p = profile::span("lang_item_query");
        let lang_items = db.crate_lang_items(start_crate);
        let start_crate_target = lang_items.items.get(&item);
        if let Some(&target) = start_crate_target {
            return Some(target);
        }
        db.crate_graph()[start_crate]
            .dependencies
            .iter()
            .find_map(|dep| db.lang_item(dep.crate_id, item.clone()))
    }

    fn collect_lang_item<T>(
        &mut self,
        db: &dyn DefDatabase,
        item: T,
        constructor: fn(T) -> LangItemTarget,
    ) where
        T: Into<AttrDefId> + Copy,
    {
        let _p = profile::span("collect_lang_item");
        if let Some(lang_item_name) = lang_attr(db, item) {
            self.items.entry(lang_item_name).or_insert_with(|| constructor(item));
        }
    }
}

pub fn lang_attr(db: &dyn DefDatabase, item: impl Into<AttrDefId> + Copy) -> Option<SmolStr> {
    let attrs = db.attrs(item.into());
    attrs.by_key("lang").string_value().cloned()
}
