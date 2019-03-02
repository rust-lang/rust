use std::sync::Arc;

use rustc_hash::FxHashMap;
use ra_arena::Arena;

use crate::{
    Function, Module, Struct, Enum, Const, Static, Trait, TypeAlias,
    Crate, PersistentHirDatabase, HirFileId, Name, Path,
    KnownName,
    nameres::{Resolution, PerNs, ModuleDef, ReachedFixedPoint},
    ids::{AstItemDef, LocationCtx, MacroCallLoc, SourceItemId, MacroCallId},
    module_tree::resolve_module_declaration,
};

use super::{CrateDefMap, ModuleId, ModuleData, raw};

#[allow(unused)]
pub(crate) fn crate_def_map_query(
    db: &impl PersistentHirDatabase,
    krate: Crate,
) -> Arc<CrateDefMap> {
    let mut modules: Arena<ModuleId, ModuleData> = Arena::default();
    let root = modules.alloc(ModuleData::default());
    let mut collector = DefCollector {
        db,
        krate,
        def_map: CrateDefMap { modules, root },
        unresolved_imports: Vec::new(),
        unexpanded_macros: Vec::new(),
        global_macro_scope: FxHashMap::default(),
    };
    collector.collect();
    let def_map = collector.finish();
    Arc::new(def_map)
}

/// Walks the tree of module recursively
struct DefCollector<DB> {
    db: DB,
    krate: Crate,
    def_map: CrateDefMap,
    unresolved_imports: Vec<(ModuleId, raw::Import)>,
    unexpanded_macros: Vec<(ModuleId, MacroCallId, tt::Subtree)>,
    global_macro_scope: FxHashMap<Name, mbe::MacroRules>,
}

/// Walks a single module, populating defs, imports and macros
struct ModCollector<'a, D> {
    def_collector: D,
    module_id: ModuleId,
    file_id: HirFileId,
    raw_items: &'a raw::RawItems,
}

impl<'a, DB> DefCollector<&'a DB>
where
    DB: PersistentHirDatabase,
{
    fn collect(&mut self) {
        let crate_graph = self.db.crate_graph();
        let file_id = crate_graph.crate_root(self.krate.crate_id());
        let raw_items = raw::RawItems::raw_items_query(self.db, file_id);
        let module_id = self.def_map.root;
        ModCollector {
            def_collector: &mut *self,
            module_id,
            file_id: file_id.into(),
            raw_items: &raw_items,
        }
        .collect(raw_items.items());

        // main name resolution fixed-point loop.
        let mut i = 0;
        loop {
            match (self.resolve_imports(), self.resolve_macros()) {
                (ReachedFixedPoint::Yes, ReachedFixedPoint::Yes) => break,
                _ => i += 1,
            }
            if i == 1000 {
                log::error!("diverging name resolution");
                break;
            }
        }
    }

    fn define_macro(&mut self, name: Name, tt: &tt::Subtree) {
        if let Ok(rules) = mbe::MacroRules::parse(tt) {
            self.global_macro_scope.insert(name, rules);
        }
    }

    fn alloc_module(&mut self) -> ModuleId {
        self.def_map.modules.alloc(ModuleData::default())
    }

    fn resolve_imports(&mut self) -> ReachedFixedPoint {
        // Resolves imports, filling-in module scopes
        ReachedFixedPoint::Yes
    }

    fn resolve_macros(&mut self) -> ReachedFixedPoint {
        // Resolve macros, calling into `expand_macro` to actually do the
        // expansion.
        ReachedFixedPoint::Yes
    }

    #[allow(unused)]
    fn expand_macro(&mut self, idx: usize, rules: &mbe::MacroRules) {
        let (module_id, call_id, arg) = self.unexpanded_macros.swap_remove(idx);
        if let Ok(tt) = rules.expand(&arg) {
            self.collect_macro_expansion(module_id, call_id, tt);
        }
    }

    fn collect_macro_expansion(
        &mut self,
        module_id: ModuleId,
        macro_call_id: MacroCallId,
        expansion: tt::Subtree,
    ) {
        // XXX: this **does not** go through a database, because we can't
        // identify macro_call without adding the whole state of name resolution
        // as a parameter to the query.
        //
        // So, we run the queries "manually" and we must ensure that
        // `db.hir_parse(macro_call_id)` returns the same source_file.
        let file_id: HirFileId = macro_call_id.into();
        let source_file = mbe::token_tree_to_ast_item_list(&expansion);

        let raw_items = raw::RawItems::from_source_file(&source_file, file_id);
        ModCollector { def_collector: &mut *self, file_id, module_id, raw_items: &raw_items }
            .collect(raw_items.items())
    }

    fn finish(self) -> CrateDefMap {
        self.def_map
    }
}

impl<DB> ModCollector<'_, &'_ mut DefCollector<&'_ DB>>
where
    DB: PersistentHirDatabase,
{
    fn collect(&mut self, items: &[raw::RawItem]) {
        for item in items {
            match *item {
                raw::RawItem::Module(m) => self.collect_module(&self.raw_items[m]),
                raw::RawItem::Import(import) => {
                    self.def_collector.unresolved_imports.push((self.module_id, import))
                }
                raw::RawItem::Def(def) => self.define_def(&self.raw_items[def]),
                raw::RawItem::Macro(mac) => self.collect_macro(&self.raw_items[mac]),
            }
        }
    }

    fn collect_module(&mut self, module: &raw::ModuleData) {
        match module {
            // inline module, just recurse
            raw::ModuleData::Definition { name, items } => {
                let module_id = self.push_child_module(name.clone());
                ModCollector {
                    def_collector: &mut *self.def_collector,
                    module_id,
                    file_id: self.file_id,
                    raw_items: self.raw_items,
                }
                .collect(&*items);
            }
            // out of line module, resovle, parse and recurse
            raw::ModuleData::Declaration { name } => {
                let module_id = self.push_child_module(name.clone());
                let is_root = self.def_collector.def_map.modules[self.module_id].parent.is_none();
                if let Some(file_id) =
                    resolve_module_declaration(self.def_collector.db, self.file_id, name, is_root)
                {
                    let raw_items = raw::RawItems::raw_items_query(self.def_collector.db, file_id);
                    ModCollector {
                        def_collector: &mut *self.def_collector,
                        module_id,
                        file_id: file_id.into(),
                        raw_items: &raw_items,
                    }
                    .collect(raw_items.items())
                }
            }
        }
    }

    fn push_child_module(&mut self, name: Name) -> ModuleId {
        let res = self.def_collector.alloc_module();
        self.def_collector.def_map.modules[res].parent = Some(self.module_id);
        self.def_collector.def_map.modules[self.module_id].children.insert(name, res);
        res
    }

    fn define_def(&mut self, def: &raw::DefData) {
        let module = Module { krate: self.def_collector.krate, module_id: self.module_id };
        let ctx = LocationCtx::new(self.def_collector.db, module, self.file_id.into());
        macro_rules! id {
            () => {
                AstItemDef::from_source_item_id_unchecked(ctx, def.source_item_id)
            };
        }
        let name = def.name.clone();
        let def: PerNs<ModuleDef> = match def.kind {
            raw::DefKind::Function => PerNs::values(Function { id: id!() }.into()),
            raw::DefKind::Struct => {
                let s = Struct { id: id!() }.into();
                PerNs::both(s, s)
            }
            raw::DefKind::Enum => PerNs::types(Enum { id: id!() }.into()),
            raw::DefKind::Const => PerNs::values(Const { id: id!() }.into()),
            raw::DefKind::Static => PerNs::values(Static { id: id!() }.into()),
            raw::DefKind::Trait => PerNs::types(Trait { id: id!() }.into()),
            raw::DefKind::TypeAlias => PerNs::types(TypeAlias { id: id!() }.into()),
        };
        let resolution = Resolution { def, import: None };
        self.def_collector.def_map.modules[self.module_id].scope.items.insert(name, resolution);
    }

    fn collect_macro(&mut self, mac: &raw::MacroData) {
        // Case 1: macro rules, define a macro in crate-global mutable scope
        if is_macro_rules(&mac.path) {
            if let Some(name) = &mac.name {
                self.def_collector.define_macro(name.clone(), &mac.arg)
            }
            return;
        }

        let source_item_id = SourceItemId { file_id: self.file_id, item_id: mac.source_item_id };
        let macro_call_id = MacroCallLoc {
            module: Module { krate: self.def_collector.krate, module_id: self.module_id },
            source_item_id,
        }
        .id(self.def_collector.db);

        // Case 2: try to expand macro_rules from this crate, triggering
        // recursive item collection.
        if let Some(rules) =
            mac.path.as_ident().and_then(|name| self.def_collector.global_macro_scope.get(name))
        {
            if let Ok(tt) = rules.expand(&mac.arg) {
                self.def_collector.collect_macro_expansion(self.module_id, macro_call_id, tt);
            }
            return;
        }

        // Case 3: path to a macro from another crate, expand during name resolution
        self.def_collector.unexpanded_macros.push((self.module_id, macro_call_id, mac.arg.clone()))
    }
}

fn is_macro_rules(path: &Path) -> bool {
    path.as_ident().and_then(Name::as_known_name) == Some(KnownName::MacroRules)
}
