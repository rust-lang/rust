use std::sync::Arc;

use rustc_hash::FxHashMap;
use ra_arena::Arena;
use test_utils::tested_by;

use crate::{
    Function, Module, Struct, Enum, Const, Static, Trait, TypeAlias,
    Crate, PersistentHirDatabase, HirFileId, Name, Path,
    KnownName,
    nameres::{Resolution, PerNs, ModuleDef, ReachedFixedPoint, ResolveMode},
    ids::{AstItemDef, LocationCtx, MacroCallLoc, SourceItemId, MacroCallId},
    module_tree::resolve_module_declaration,
};

use super::{CrateDefMap, ModuleId, ModuleData, raw};

#[allow(unused)]
pub(crate) fn crate_def_map_query(
    db: &impl PersistentHirDatabase,
    krate: Crate,
) -> Arc<CrateDefMap> {
    let mut def_map = {
        let edition = krate.edition(db);
        let mut modules: Arena<ModuleId, ModuleData> = Arena::default();
        let root = modules.alloc(ModuleData::default());
        CrateDefMap {
            krate,
            edition,
            extern_prelude: FxHashMap::default(),
            prelude: None,
            root,
            modules,
            public_macros: FxHashMap::default(),
        }
    };

    // populate external prelude
    for dep in krate.dependencies(db) {
        log::debug!("crate dep {:?} -> {:?}", dep.name, dep.krate);
        if let Some(module) = dep.krate.root_module(db) {
            def_map.extern_prelude.insert(dep.name.clone(), module.into());
        }
        // look for the prelude
        if def_map.prelude.is_none() {
            let item_map = db.item_map(dep.krate);
            if item_map.prelude.is_some() {
                def_map.prelude = item_map.prelude;
            }
        }
    }

    let mut collector = DefCollector {
        db,
        krate,
        def_map,
        glob_imports: FxHashMap::default(),
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
    glob_imports: FxHashMap<ModuleId, Vec<(ModuleId, raw::ImportId)>>,
    unresolved_imports: Vec<(ModuleId, raw::ImportId, raw::ImportData)>,
    unexpanded_macros: Vec<(ModuleId, MacroCallId, Path, tt::Subtree)>,
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

    fn define_macro(&mut self, name: Name, tt: &tt::Subtree, export: bool) {
        if let Ok(rules) = mbe::MacroRules::parse(tt) {
            if export {
                self.def_map.public_macros.insert(name.clone(), rules.clone());
            }
            self.global_macro_scope.insert(name, rules);
        }
    }

    fn alloc_module(&mut self) -> ModuleId {
        self.def_map.modules.alloc(ModuleData::default())
    }

    fn resolve_imports(&mut self) -> ReachedFixedPoint {
        let mut imports = std::mem::replace(&mut self.unresolved_imports, Vec::new());
        let mut resolved = Vec::new();
        imports.retain(|(module_id, import, import_data)| {
            let (def, fp) = self.resolve_import(*module_id, import_data);
            if fp == ReachedFixedPoint::Yes {
                resolved.push((*module_id, def, *import, import_data.clone()))
            }
            fp == ReachedFixedPoint::No
        });
        self.unresolved_imports = imports;
        // Resolves imports, filling-in module scopes
        let result =
            if resolved.is_empty() { ReachedFixedPoint::Yes } else { ReachedFixedPoint::No };
        for (module_id, def, import, import_data) in resolved {
            self.record_resolved_import(module_id, def, import, &import_data)
        }
        result
    }

    fn resolve_import(
        &mut self,
        module_id: ModuleId,
        import: &raw::ImportData,
    ) -> (PerNs<ModuleDef>, ReachedFixedPoint) {
        log::debug!("resolving import: {:?} ({:?})", import, self.def_map.edition);
        if import.is_extern_crate {
            let res = self.def_map.resolve_name_in_extern_prelude(
                &import
                    .path
                    .as_ident()
                    .expect("extern crate should have been desugared to one-element path"),
            );
            // FIXME: why do we return No here?
            (res, if res.is_none() { ReachedFixedPoint::No } else { ReachedFixedPoint::Yes })
        } else {
            let res =
                self.def_map.resolve_path_fp(self.db, ResolveMode::Import, module_id, &import.path);

            (res.resolved_def, res.reached_fixedpoint)
        }
    }

    fn record_resolved_import(
        &mut self,
        module_id: ModuleId,
        def: PerNs<ModuleDef>,
        import_id: raw::ImportId,
        import: &raw::ImportData,
    ) {
        if import.is_glob {
            log::debug!("glob import: {:?}", import);
            match def.take_types() {
                Some(ModuleDef::Module(m)) => {
                    if import.is_prelude {
                        tested_by!(std_prelude);
                        self.def_map.prelude = Some(m);
                    } else if m.krate != self.krate {
                        tested_by!(glob_across_crates);
                        // glob import from other crate => we can just import everything once
                        let item_map = self.db.item_map(m.krate);
                        let scope = &item_map[m.module_id];
                        let items = scope
                            .items
                            .iter()
                            .map(|(name, res)| (name.clone(), res.clone()))
                            .collect::<Vec<_>>();
                        self.update(module_id, Some(import_id), &items);
                    } else {
                        // glob import from same crate => we do an initial
                        // import, and then need to propagate any further
                        // additions
                        let scope = &self.def_map[m.module_id];
                        let items = scope
                            .items
                            .iter()
                            .map(|(name, res)| (name.clone(), res.clone()))
                            .collect::<Vec<_>>();
                        self.update(module_id, Some(import_id), &items);
                        // record the glob import in case we add further items
                        self.glob_imports
                            .entry(m.module_id)
                            .or_default()
                            .push((module_id, import_id));
                    }
                }
                Some(ModuleDef::Enum(e)) => {
                    tested_by!(glob_enum);
                    // glob import from enum => just import all the variants
                    let variants = e.variants(self.db);
                    let resolutions = variants
                        .into_iter()
                        .filter_map(|variant| {
                            let res = Resolution {
                                def: PerNs::both(variant.into(), variant.into()),
                                import: Some(import_id),
                            };
                            let name = variant.name(self.db)?;
                            Some((name, res))
                        })
                        .collect::<Vec<_>>();
                    self.update(module_id, Some(import_id), &resolutions);
                }
                Some(d) => {
                    log::debug!("glob import {:?} from non-module/enum {:?}", import, d);
                }
                None => {
                    log::debug!("glob import {:?} didn't resolve as type", import);
                }
            }
        } else {
            let last_segment = import.path.segments.last().unwrap();
            let name = import.alias.clone().unwrap_or_else(|| last_segment.name.clone());
            log::debug!("resolved import {:?} ({:?}) to {:?}", name, import, def);

            // extern crates in the crate root are special-cased to insert entries into the extern prelude: rust-lang/rust#54658
            if let Some(root_module) = self.krate.root_module(self.db) {
                if import.is_extern_crate && module_id == root_module.module_id {
                    if let Some(def) = def.take_types() {
                        self.def_map.extern_prelude.insert(name.clone(), def);
                    }
                }
            }
            let resolution = Resolution { def, import: Some(import_id) };
            self.update(module_id, None, &[(name, resolution)]);
        }
    }

    fn update(
        &mut self,
        module_id: ModuleId,
        import: Option<raw::ImportId>,
        resolutions: &[(Name, Resolution)],
    ) {
        self.update_recursive(module_id, import, resolutions, 0)
    }

    fn update_recursive(
        &mut self,
        module_id: ModuleId,
        import: Option<raw::ImportId>,
        resolutions: &[(Name, Resolution)],
        depth: usize,
    ) {
        if depth > 100 {
            // prevent stack overflows (but this shouldn't be possible)
            panic!("infinite recursion in glob imports!");
        }
        let module_items = &mut self.def_map.modules[module_id].scope;
        let mut changed = false;
        for (name, res) in resolutions {
            let existing = module_items.items.entry(name.clone()).or_default();
            if existing.def.types.is_none() && res.def.types.is_some() {
                existing.def.types = res.def.types;
                existing.import = import.or(res.import);
                changed = true;
            }
            if existing.def.values.is_none() && res.def.values.is_some() {
                existing.def.values = res.def.values;
                existing.import = import.or(res.import);
                changed = true;
            }
        }
        if !changed {
            return;
        }
        let glob_imports = self
            .glob_imports
            .get(&module_id)
            .into_iter()
            .flat_map(|v| v.iter())
            .cloned()
            .collect::<Vec<_>>();
        for (glob_importing_module, glob_import) in glob_imports {
            // We pass the glob import so that the tracked import in those modules is that glob import
            self.update_recursive(glob_importing_module, Some(glob_import), resolutions, depth + 1);
        }
    }

    // XXX: this is just a pile of hacks now, because `PerNs` does not handle
    // macro namespace.
    fn resolve_macros(&mut self) -> ReachedFixedPoint {
        let mut macros = std::mem::replace(&mut self.unexpanded_macros, Vec::new());
        let mut resolved = Vec::new();
        macros.retain(|(module_id, call_id, path, tt)| {
            if path.segments.len() != 2 {
                return true;
            }
            let crate_name = &path.segments[0].name;
            let krate = match self.def_map.resolve_name_in_extern_prelude(crate_name).take_types() {
                Some(ModuleDef::Module(m)) => m.krate(self.db),
                _ => return true,
            };
            let krate = match krate {
                Some(it) => it,
                _ => return true,
            };
            // FIXME: this should be a proper query
            let def_map = crate_def_map_query(self.db, krate);
            let rules = def_map.public_macros.get(&path.segments[1].name).cloned();
            resolved.push((*module_id, *call_id, rules, tt.clone()));
            false
        });
        let res = if resolved.is_empty() { ReachedFixedPoint::Yes } else { ReachedFixedPoint::No };

        for (module_id, macro_call_id, rules, arg) in resolved {
            if let Some(rules) = rules {
                if let Ok(tt) = rules.expand(&arg) {
                    self.collect_macro_expansion(module_id, macro_call_id, tt);
                }
            }
        }
        res
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
                raw::RawItem::Import(import) => self.def_collector.unresolved_imports.push((
                    self.module_id,
                    import,
                    self.raw_items[import].clone(),
                )),
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
        self.def_collector.update(self.module_id, None, &[(name, resolution)])
    }

    fn collect_macro(&mut self, mac: &raw::MacroData) {
        // Case 1: macro rules, define a macro in crate-global mutable scope
        if is_macro_rules(&mac.path) {
            if let Some(name) = &mac.name {
                self.def_collector.define_macro(name.clone(), &mac.arg, mac.export)
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
        self.def_collector.unexpanded_macros.push((
            self.module_id,
            macro_call_id,
            mac.path.clone(),
            mac.arg.clone(),
        ))
    }
}

fn is_macro_rules(path: &Path) -> bool {
    path.as_ident().and_then(Name::as_known_name) == Some(KnownName::MacroRules)
}
