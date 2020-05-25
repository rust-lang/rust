//! Collects lang items: items marked with `#[lang = "..."]` attribute.
//!
//! This attribute to tell the compiler about semi built-in std library
//! features, such as Fn family of traits.
use std::sync::Arc;

use ra_prof::profile;
use ra_syntax::SmolStr;
use rustc_hash::FxHashMap;

use crate::{
    db::DefDatabase, AdtId, AttrDefId, CrateId, EnumId, FunctionId, ImplId, ModuleDefId, ModuleId,
    StaticId, StructId, TraitId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LangItemTarget {
    EnumId(EnumId),
    FunctionId(FunctionId),
    ImplDefId(ImplId),
    StaticId(StaticId),
    StructId(StructId),
    TraitId(TraitId),
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
        let _p = profile("crate_lang_items_query");

        let mut lang_items = LangItems::default();

        let crate_def_map = db.crate_def_map(krate);

        crate_def_map
            .modules
            .iter()
            .filter_map(|(local_id, _)| db.module_lang_items(ModuleId { krate, local_id }))
            .for_each(|it| lang_items.items.extend(it.items.iter().map(|(k, v)| (k.clone(), *v))));

        Arc::new(lang_items)
    }

    pub(crate) fn module_lang_items_query(
        db: &dyn DefDatabase,
        module: ModuleId,
    ) -> Option<Arc<LangItems>> {
        let _p = profile("module_lang_items_query");
        let mut lang_items = LangItems::default();
        lang_items.collect_lang_items(db, module);
        if lang_items.items.is_empty() {
            None
        } else {
            Some(Arc::new(lang_items))
        }
    }

    /// Salsa query. Look for a lang item, starting from the specified crate and recursively
    /// traversing its dependencies.
    pub(crate) fn lang_item_query(
        db: &dyn DefDatabase,
        start_crate: CrateId,
        item: SmolStr,
    ) -> Option<LangItemTarget> {
        let _p = profile("lang_item_query");
        let lang_items = db.crate_lang_items(start_crate);
        let start_crate_target = lang_items.items.get(&item);
        if let Some(target) = start_crate_target {
            return Some(*target);
        }
        db.crate_graph()[start_crate]
            .dependencies
            .iter()
            .find_map(|dep| db.lang_item(dep.crate_id, item.clone()))
    }

    fn collect_lang_items(&mut self, db: &dyn DefDatabase, module: ModuleId) {
        // Look for impl targets
        let def_map = db.crate_def_map(module.krate);
        let module_data = &def_map[module.local_id];
        for impl_def in module_data.scope.impls() {
            self.collect_lang_item(db, impl_def, LangItemTarget::ImplDefId)
        }

        for def in module_data.scope.declarations() {
            match def {
                ModuleDefId::TraitId(trait_) => {
                    self.collect_lang_item(db, trait_, LangItemTarget::TraitId)
                }
                ModuleDefId::AdtId(AdtId::EnumId(e)) => {
                    self.collect_lang_item(db, e, LangItemTarget::EnumId)
                }
                ModuleDefId::AdtId(AdtId::StructId(s)) => {
                    self.collect_lang_item(db, s, LangItemTarget::StructId)
                }
                ModuleDefId::FunctionId(f) => {
                    self.collect_lang_item(db, f, LangItemTarget::FunctionId)
                }
                ModuleDefId::StaticId(s) => self.collect_lang_item(db, s, LangItemTarget::StaticId),
                _ => {}
            }
        }
    }

    fn collect_lang_item<T>(
        &mut self,
        db: &dyn DefDatabase,
        item: T,
        constructor: fn(T) -> LangItemTarget,
    ) where
        T: Into<AttrDefId> + Copy,
    {
        if let Some(lang_item_name) = lang_attr(db, item) {
            self.items.entry(lang_item_name).or_insert_with(|| constructor(item));
        }
    }
}

pub fn lang_attr(db: &dyn DefDatabase, item: impl Into<AttrDefId> + Copy) -> Option<SmolStr> {
    let attrs = db.attrs(item.into());
    attrs.by_key("lang").string_value().cloned()
}
