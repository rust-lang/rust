//! FIXME: write short doc here

use std::sync::Arc;

use hir_def::{AdtId, AttrDefId, ModuleDefId};
use ra_syntax::SmolStr;
use rustc_hash::FxHashMap;

use crate::{
    db::{AstDatabase, DefDatabase, HirDatabase},
    Crate, Enum, Function, ImplBlock, Module, Static, Struct, Trait,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LangItemTarget {
    Enum(Enum),
    Function(Function),
    ImplBlock(ImplBlock),
    Static(Static),
    Struct(Struct),
    Trait(Trait),
}

impl LangItemTarget {
    pub(crate) fn krate(&self, db: &impl HirDatabase) -> Option<Crate> {
        Some(match self {
            LangItemTarget::Enum(e) => e.module(db).krate(),
            LangItemTarget::Function(f) => f.module(db).krate(),
            LangItemTarget::ImplBlock(i) => i.krate(db),
            LangItemTarget::Static(s) => s.module(db).krate(),
            LangItemTarget::Struct(s) => s.module(db).krate(),
            LangItemTarget::Trait(t) => t.module(db).krate(),
        })
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct LangItems {
    items: FxHashMap<SmolStr, LangItemTarget>,
}

impl LangItems {
    pub fn target<'a>(&'a self, item: &str) -> Option<&'a LangItemTarget> {
        self.items.get(item)
    }

    /// Salsa query. This will look for lang items in a specific crate.
    pub(crate) fn crate_lang_items_query(
        db: &(impl DefDatabase + AstDatabase),
        krate: Crate,
    ) -> Arc<LangItems> {
        let mut lang_items = LangItems::default();

        if let Some(module) = krate.root_module(db) {
            lang_items.collect_lang_items_recursive(db, module);
        }

        Arc::new(lang_items)
    }

    pub(crate) fn module_lang_items_query(
        db: &(impl DefDatabase + AstDatabase),
        module: Module,
    ) -> Option<Arc<LangItems>> {
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
        db: &impl DefDatabase,
        start_crate: Crate,
        item: SmolStr,
    ) -> Option<LangItemTarget> {
        let lang_items = db.crate_lang_items(start_crate);
        let start_crate_target = lang_items.items.get(&item);
        if let Some(target) = start_crate_target {
            Some(*target)
        } else {
            for dep in start_crate.dependencies(db) {
                let dep_crate = dep.krate;
                let dep_target = db.lang_item(dep_crate, item.clone());
                if dep_target.is_some() {
                    return dep_target;
                }
            }
            None
        }
    }

    fn collect_lang_items(&mut self, db: &(impl DefDatabase + AstDatabase), module: Module) {
        // Look for impl targets
        let def_map = db.crate_def_map(module.id.krate);
        let module_data = &def_map[module.id.module_id];
        for &impl_block in module_data.impls.iter() {
            self.collect_lang_item(db, impl_block, LangItemTarget::ImplBlock)
        }

        for def in module_data.scope.declarations() {
            match def {
                ModuleDefId::TraitId(trait_) => {
                    self.collect_lang_item(db, trait_, LangItemTarget::Trait)
                }
                ModuleDefId::AdtId(AdtId::EnumId(e)) => {
                    self.collect_lang_item(db, e, LangItemTarget::Enum)
                }
                ModuleDefId::AdtId(AdtId::StructId(s)) => {
                    self.collect_lang_item(db, s, LangItemTarget::Struct)
                }
                ModuleDefId::FunctionId(f) => {
                    self.collect_lang_item(db, f, LangItemTarget::Function)
                }
                ModuleDefId::StaticId(s) => self.collect_lang_item(db, s, LangItemTarget::Static),
                _ => {}
            }
        }
    }

    fn collect_lang_items_recursive(
        &mut self,
        db: &(impl DefDatabase + AstDatabase),
        module: Module,
    ) {
        if let Some(module_lang_items) = db.module_lang_items(module) {
            self.items.extend(module_lang_items.items.iter().map(|(k, v)| (k.clone(), *v)))
        }

        // Look for lang items in the children
        for child in module.children(db) {
            self.collect_lang_items_recursive(db, child);
        }
    }

    fn collect_lang_item<T, D>(
        &mut self,
        db: &(impl DefDatabase + AstDatabase),
        item: T,
        constructor: fn(D) -> LangItemTarget,
    ) where
        T: Into<AttrDefId> + Copy,
        D: From<T>,
    {
        let attrs = db.attrs(item.into());
        if let Some(lang_item_name) = attrs.find_string_value("lang") {
            self.items.entry(lang_item_name).or_insert_with(|| constructor(D::from(item)));
        }
    }
}
