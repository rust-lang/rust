use std::sync::Arc;
use rustc_hash::FxHashMap;

use ra_syntax::{SmolStr, TreeArc, ast::AttrsOwner};

use crate::{
    Crate, DefDatabase, Enum, Function, HirDatabase, ImplBlock, Module,
    Static, Struct, Trait, ModuleDef, AstDatabase, HasSource
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
        match self {
            LangItemTarget::Enum(e) => e.module(db).krate(db),
            LangItemTarget::Function(f) => f.module(db).krate(db),
            LangItemTarget::ImplBlock(i) => i.module().krate(db),
            LangItemTarget::Static(s) => s.module(db).krate(db),
            LangItemTarget::Struct(s) => s.module(db).krate(db),
            LangItemTarget::Trait(t) => t.module(db).krate(db),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LangItems {
    items: FxHashMap<SmolStr, LangItemTarget>,
}

impl LangItems {
    pub fn target<'a>(&'a self, item: &str) -> Option<&'a LangItemTarget> {
        self.items.get(item)
    }

    /// Salsa query. This will look for lang items in a specific crate.
    pub(crate) fn lang_items_query(
        db: &(impl DefDatabase + AstDatabase),
        krate: Crate,
    ) -> Arc<LangItems> {
        let mut lang_items = LangItems { items: FxHashMap::default() };

        if let Some(module) = krate.root_module(db) {
            lang_items.collect_lang_items_recursive(db, &module);
        }

        Arc::new(lang_items)
    }

    /// Salsa query. Look for a lang item, starting from the specified crate and recursively
    /// traversing its dependencies.
    pub(crate) fn lang_item_query(
        db: &impl DefDatabase,
        start_crate: Crate,
        item: SmolStr,
    ) -> Option<LangItemTarget> {
        let lang_items = db.lang_items(start_crate);
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

    fn collect_lang_items_recursive(
        &mut self,
        db: &(impl DefDatabase + AstDatabase),
        module: &Module,
    ) {
        // Look for impl targets
        let (impl_blocks, source_map) = db.impls_in_module_with_source_map(module.clone());
        let source = module.definition_source(db).ast;
        for (impl_id, _) in impl_blocks.impls.iter() {
            let impl_block = source_map.get(&source, impl_id);
            if let Some(lang_item_name) = lang_item_name(&*impl_block) {
                let imp = ImplBlock::from_id(*module, impl_id);
                self.items.entry(lang_item_name).or_insert_with(|| LangItemTarget::ImplBlock(imp));
            }
        }

        for def in module.declarations(db) {
            match def {
                ModuleDef::Trait(trait_) => {
                    self.collect_lang_item(db, trait_, LangItemTarget::Trait)
                }
                ModuleDef::Enum(e) => self.collect_lang_item(db, e, LangItemTarget::Enum),
                ModuleDef::Struct(s) => self.collect_lang_item(db, s, LangItemTarget::Struct),
                ModuleDef::Function(f) => self.collect_lang_item(db, f, LangItemTarget::Function),
                ModuleDef::Static(s) => self.collect_lang_item(db, s, LangItemTarget::Static),
                _ => {}
            }
        }

        // Look for lang items in the children
        for child in module.children(db) {
            self.collect_lang_items_recursive(db, &child);
        }
    }

    fn collect_lang_item<T, N>(
        &mut self,
        db: &(impl DefDatabase + AstDatabase),
        item: T,
        constructor: fn(T) -> LangItemTarget,
    ) where
        T: Copy + HasSource<Ast = TreeArc<N>>,
        N: AttrsOwner,
    {
        let node = item.source(db).ast;
        if let Some(lang_item_name) = lang_item_name(&*node) {
            self.items.entry(lang_item_name).or_insert(constructor(item));
        }
    }
}

fn lang_item_name<T: AttrsOwner>(node: &T) -> Option<SmolStr> {
    node.attrs()
        .filter_map(|a| a.as_key_value())
        .filter(|(key, _)| key == "lang")
        .map(|(_, val)| val)
        .nth(0)
}
