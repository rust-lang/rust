//! The core of the module-level name resolution algorithm.
//!
//! `DefCollector::collect` contains the fixed-point iteration loop which
//! resolves imports and expands macros.

use base_db::{CrateId, FileId, ProcMacroId};
use cfg::CfgOptions;
use hir_expand::{
    ast_id_map::FileAstId,
    builtin_derive::find_builtin_derive,
    builtin_macro::find_builtin_macro,
    name::{name, AsName, Name},
    proc_macro::ProcMacroExpander,
    HirFileId, MacroCallId, MacroDefId, MacroDefKind,
};
use rustc_hash::FxHashMap;
use syntax::ast;
use test_utils::mark;

use crate::{
    attr::Attrs,
    db::DefDatabase,
    item_scope::{ImportType, PerNsGlobImports},
    item_tree::{
        self, FileItemTreeId, ItemTree, ItemTreeId, MacroCall, Mod, ModItem, ModKind, StructDefKind,
    },
    nameres::{
        diagnostics::DefDiagnostic, mod_resolution::ModDir, path_resolution::ReachedFixedPoint,
        BuiltinShadowMode, CrateDefMap, ModuleData, ModuleOrigin, ResolveMode,
    },
    path::{ImportAlias, ModPath, PathKind},
    per_ns::PerNs,
    visibility::{RawVisibility, Visibility},
    AdtId, AsMacroCall, AstId, AstIdWithPath, ConstLoc, ContainerId, EnumLoc, EnumVariantId,
    FunctionLoc, ImplLoc, Intern, LocalModuleId, ModuleDefId, ModuleId, StaticLoc, StructLoc,
    TraitLoc, TypeAliasLoc, UnionLoc,
};

const GLOB_RECURSION_LIMIT: usize = 100;
const EXPANSION_DEPTH_LIMIT: usize = 128;
const FIXED_POINT_LIMIT: usize = 8192;

pub(super) fn collect_defs(db: &dyn DefDatabase, mut def_map: CrateDefMap) -> CrateDefMap {
    let crate_graph = db.crate_graph();

    // populate external prelude
    for dep in &crate_graph[def_map.krate].dependencies {
        log::debug!("crate dep {:?} -> {:?}", dep.name, dep.crate_id);
        let dep_def_map = db.crate_def_map(dep.crate_id);
        def_map.extern_prelude.insert(
            dep.as_name(),
            ModuleId { krate: dep.crate_id, local_id: dep_def_map.root }.into(),
        );

        // look for the prelude
        // If the dependency defines a prelude, we overwrite an already defined
        // prelude. This is necessary to import the "std" prelude if a crate
        // depends on both "core" and "std".
        if dep_def_map.prelude.is_some() {
            def_map.prelude = dep_def_map.prelude;
        }
    }

    let cfg_options = &crate_graph[def_map.krate].cfg_options;
    let proc_macros = &crate_graph[def_map.krate].proc_macro;
    let proc_macros = proc_macros
        .iter()
        .enumerate()
        .map(|(idx, it)| {
            // FIXME: a hacky way to create a Name from string.
            let name = tt::Ident { text: it.name.clone(), id: tt::TokenId::unspecified() };
            (name.as_name(), ProcMacroExpander::new(def_map.krate, ProcMacroId(idx as u32)))
        })
        .collect();

    let mut collector = DefCollector {
        db,
        def_map,
        glob_imports: FxHashMap::default(),
        unresolved_imports: Vec::new(),
        resolved_imports: Vec::new(),

        unexpanded_macros: Vec::new(),
        unexpanded_attribute_macros: Vec::new(),
        mod_dirs: FxHashMap::default(),
        cfg_options,
        proc_macros,
        from_glob_import: Default::default(),
    };
    collector.collect();
    collector.finish()
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum PartialResolvedImport {
    /// None of any namespaces is resolved
    Unresolved,
    /// One of namespaces is resolved
    Indeterminate(PerNs),
    /// All namespaces are resolved, OR it is came from other crate
    Resolved(PerNs),
}

impl PartialResolvedImport {
    fn namespaces(&self) -> PerNs {
        match self {
            PartialResolvedImport::Unresolved => PerNs::none(),
            PartialResolvedImport::Indeterminate(ns) => *ns,
            PartialResolvedImport::Resolved(ns) => *ns,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Import {
    pub path: ModPath,
    pub alias: Option<ImportAlias>,
    pub visibility: RawVisibility,
    pub is_glob: bool,
    pub is_prelude: bool,
    pub is_extern_crate: bool,
    pub is_macro_use: bool,
}

impl Import {
    fn from_use(tree: &ItemTree, id: FileItemTreeId<item_tree::Import>) -> Self {
        let it = &tree[id];
        let visibility = &tree[it.visibility];
        Self {
            path: it.path.clone(),
            alias: it.alias.clone(),
            visibility: visibility.clone(),
            is_glob: it.is_glob,
            is_prelude: it.is_prelude,
            is_extern_crate: false,
            is_macro_use: false,
        }
    }

    fn from_extern_crate(tree: &ItemTree, id: FileItemTreeId<item_tree::ExternCrate>) -> Self {
        let it = &tree[id];
        let visibility = &tree[it.visibility];
        Self {
            path: it.path.clone(),
            alias: it.alias.clone(),
            visibility: visibility.clone(),
            is_glob: false,
            is_prelude: false,
            is_extern_crate: true,
            is_macro_use: it.is_macro_use,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ImportDirective {
    module_id: LocalModuleId,
    import: Import,
    status: PartialResolvedImport,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MacroDirective {
    module_id: LocalModuleId,
    ast_id: AstIdWithPath<ast::MacroCall>,
    legacy: Option<MacroCallId>,
    depth: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DeriveDirective {
    module_id: LocalModuleId,
    ast_id: AstIdWithPath<ast::Item>,
}

struct DefData<'a> {
    id: ModuleDefId,
    name: &'a Name,
    visibility: &'a RawVisibility,
    has_constructor: bool,
}

/// Walks the tree of module recursively
struct DefCollector<'a> {
    db: &'a dyn DefDatabase,
    def_map: CrateDefMap,
    glob_imports: FxHashMap<LocalModuleId, Vec<(LocalModuleId, Visibility)>>,
    unresolved_imports: Vec<ImportDirective>,
    resolved_imports: Vec<ImportDirective>,
    unexpanded_macros: Vec<MacroDirective>,
    unexpanded_attribute_macros: Vec<DeriveDirective>,
    mod_dirs: FxHashMap<LocalModuleId, ModDir>,
    cfg_options: &'a CfgOptions,
    proc_macros: Vec<(Name, ProcMacroExpander)>,
    from_glob_import: PerNsGlobImports,
}

impl DefCollector<'_> {
    fn collect(&mut self) {
        let file_id = self.db.crate_graph()[self.def_map.krate].root_file_id;
        let item_tree = self.db.item_tree(file_id.into());
        let module_id = self.def_map.root;
        self.def_map.modules[module_id].origin = ModuleOrigin::CrateRoot { definition: file_id };
        ModCollector {
            def_collector: &mut *self,
            macro_depth: 0,
            module_id,
            file_id: file_id.into(),
            item_tree: &item_tree,
            mod_dir: ModDir::root(),
        }
        .collect(item_tree.top_level_items());

        // main name resolution fixed-point loop.
        let mut i = 0;
        loop {
            self.db.check_canceled();
            self.resolve_imports();

            match self.resolve_macros() {
                ReachedFixedPoint::Yes => break,
                ReachedFixedPoint::No => i += 1,
            }
            if i == FIXED_POINT_LIMIT {
                log::error!("name resolution is stuck");
                break;
            }
        }

        // Resolve all indeterminate resolved imports again
        // As some of the macros will expand newly import shadowing partial resolved imports
        // FIXME: We maybe could skip this, if we handle the Indetermine imports in `resolve_imports`
        // correctly
        let partial_resolved = self.resolved_imports.iter().filter_map(|directive| {
            if let PartialResolvedImport::Indeterminate(_) = directive.status {
                let mut directive = directive.clone();
                directive.status = PartialResolvedImport::Unresolved;
                Some(directive)
            } else {
                None
            }
        });
        self.unresolved_imports.extend(partial_resolved);
        self.resolve_imports();

        let unresolved_imports = std::mem::replace(&mut self.unresolved_imports, Vec::new());
        // show unresolved imports in completion, etc
        for directive in unresolved_imports {
            self.record_resolved_import(&directive)
        }

        // Record proc-macros
        self.collect_proc_macro();
    }

    fn collect_proc_macro(&mut self) {
        let proc_macros = std::mem::take(&mut self.proc_macros);
        for (name, expander) in proc_macros {
            let krate = self.def_map.krate;

            let macro_id = MacroDefId {
                ast_id: None,
                krate: Some(krate),
                kind: MacroDefKind::CustomDerive(expander),
                local_inner: false,
            };

            self.define_proc_macro(name.clone(), macro_id);
        }
    }

    /// Define a macro with `macro_rules`.
    ///
    /// It will define the macro in legacy textual scope, and if it has `#[macro_export]`,
    /// then it is also defined in the root module scope.
    /// You can `use` or invoke it by `crate::macro_name` anywhere, before or after the definition.
    ///
    /// It is surprising that the macro will never be in the current module scope.
    /// These code fails with "unresolved import/macro",
    /// ```rust,compile_fail
    /// mod m { macro_rules! foo { () => {} } }
    /// use m::foo as bar;
    /// ```
    ///
    /// ```rust,compile_fail
    /// macro_rules! foo { () => {} }
    /// self::foo!();
    /// crate::foo!();
    /// ```
    ///
    /// Well, this code compiles, because the plain path `foo` in `use` is searched
    /// in the legacy textual scope only.
    /// ```rust
    /// macro_rules! foo { () => {} }
    /// use foo as bar;
    /// ```
    fn define_macro(
        &mut self,
        module_id: LocalModuleId,
        name: Name,
        macro_: MacroDefId,
        export: bool,
    ) {
        // Textual scoping
        self.define_legacy_macro(module_id, name.clone(), macro_);

        // Module scoping
        // In Rust, `#[macro_export]` macros are unconditionally visible at the
        // crate root, even if the parent modules is **not** visible.
        if export {
            self.update(
                self.def_map.root,
                &[(Some(name), PerNs::macros(macro_, Visibility::Public))],
                Visibility::Public,
                ImportType::Named,
            );
        }
    }

    /// Define a legacy textual scoped macro in module
    ///
    /// We use a map `legacy_macros` to store all legacy textual scoped macros visible per module.
    /// It will clone all macros from parent legacy scope, whose definition is prior to
    /// the definition of current module.
    /// And also, `macro_use` on a module will import all legacy macros visible inside to
    /// current legacy scope, with possible shadowing.
    fn define_legacy_macro(&mut self, module_id: LocalModuleId, name: Name, mac: MacroDefId) {
        // Always shadowing
        self.def_map.modules[module_id].scope.define_legacy_macro(name, mac);
    }

    /// Define a proc macro
    ///
    /// A proc macro is similar to normal macro scope, but it would not visiable in legacy textual scoped.
    /// And unconditionally exported.
    fn define_proc_macro(&mut self, name: Name, macro_: MacroDefId) {
        self.update(
            self.def_map.root,
            &[(Some(name), PerNs::macros(macro_, Visibility::Public))],
            Visibility::Public,
            ImportType::Named,
        );
    }

    /// Import macros from `#[macro_use] extern crate`.
    fn import_macros_from_extern_crate(
        &mut self,
        current_module_id: LocalModuleId,
        import: &item_tree::ExternCrate,
    ) {
        log::debug!(
            "importing macros from extern crate: {:?} ({:?})",
            import,
            self.def_map.edition,
        );

        let res = self.def_map.resolve_name_in_extern_prelude(
            &import
                .path
                .as_ident()
                .expect("extern crate should have been desugared to one-element path"),
        );

        if let Some(ModuleDefId::ModuleId(m)) = res.take_types() {
            mark::hit!(macro_rules_from_other_crates_are_visible_with_macro_use);
            self.import_all_macros_exported(current_module_id, m.krate);
        }
    }

    /// Import all exported macros from another crate
    ///
    /// Exported macros are just all macros in the root module scope.
    /// Note that it contains not only all `#[macro_export]` macros, but also all aliases
    /// created by `use` in the root module, ignoring the visibility of `use`.
    fn import_all_macros_exported(&mut self, current_module_id: LocalModuleId, krate: CrateId) {
        let def_map = self.db.crate_def_map(krate);
        for (name, def) in def_map[def_map.root].scope.macros() {
            // `macro_use` only bring things into legacy scope.
            self.define_legacy_macro(current_module_id, name.clone(), def);
        }
    }

    /// Import resolution
    ///
    /// This is a fix point algorithm. We resolve imports until no forward
    /// progress in resolving imports is made
    fn resolve_imports(&mut self) {
        let mut n_previous_unresolved = self.unresolved_imports.len() + 1;

        while self.unresolved_imports.len() < n_previous_unresolved {
            n_previous_unresolved = self.unresolved_imports.len();
            let imports = std::mem::replace(&mut self.unresolved_imports, Vec::new());
            for mut directive in imports {
                directive.status = self.resolve_import(directive.module_id, &directive.import);
                match directive.status {
                    PartialResolvedImport::Indeterminate(_) => {
                        self.record_resolved_import(&directive);
                        // FIXME: For avoid performance regression,
                        // we consider an imported resolved if it is indeterminate (i.e not all namespace resolved)
                        self.resolved_imports.push(directive)
                    }
                    PartialResolvedImport::Resolved(_) => {
                        self.record_resolved_import(&directive);
                        self.resolved_imports.push(directive)
                    }
                    PartialResolvedImport::Unresolved => {
                        self.unresolved_imports.push(directive);
                    }
                }
            }
        }
    }

    fn resolve_import(&self, module_id: LocalModuleId, import: &Import) -> PartialResolvedImport {
        log::debug!("resolving import: {:?} ({:?})", import, self.def_map.edition);
        if import.is_extern_crate {
            let res = self.def_map.resolve_name_in_extern_prelude(
                &import
                    .path
                    .as_ident()
                    .expect("extern crate should have been desugared to one-element path"),
            );
            PartialResolvedImport::Resolved(res)
        } else {
            let res = self.def_map.resolve_path_fp_with_macro(
                self.db,
                ResolveMode::Import,
                module_id,
                &import.path,
                BuiltinShadowMode::Module,
            );

            let def = res.resolved_def;
            if res.reached_fixedpoint == ReachedFixedPoint::No || def.is_none() {
                return PartialResolvedImport::Unresolved;
            }

            if let Some(krate) = res.krate {
                if krate != self.def_map.krate {
                    return PartialResolvedImport::Resolved(def);
                }
            }

            // Check whether all namespace is resolved
            if def.take_types().is_some()
                && def.take_values().is_some()
                && def.take_macros().is_some()
            {
                PartialResolvedImport::Resolved(def)
            } else {
                PartialResolvedImport::Indeterminate(def)
            }
        }
    }

    fn record_resolved_import(&mut self, directive: &ImportDirective) {
        let module_id = directive.module_id;
        let import = &directive.import;
        let def = directive.status.namespaces();
        let vis = self
            .def_map
            .resolve_visibility(self.db, module_id, &directive.import.visibility)
            .unwrap_or(Visibility::Public);

        if import.is_glob {
            log::debug!("glob import: {:?}", import);
            match def.take_types() {
                Some(ModuleDefId::ModuleId(m)) => {
                    if import.is_prelude {
                        mark::hit!(std_prelude);
                        self.def_map.prelude = Some(m);
                    } else if m.krate != self.def_map.krate {
                        mark::hit!(glob_across_crates);
                        // glob import from other crate => we can just import everything once
                        let item_map = self.db.crate_def_map(m.krate);
                        let scope = &item_map[m.local_id].scope;

                        // Module scoped macros is included
                        let items = scope
                            .resolutions()
                            // only keep visible names...
                            .map(|(n, res)| {
                                (n, res.filter_visibility(|v| v.is_visible_from_other_crate()))
                            })
                            .filter(|(_, res)| !res.is_none())
                            .collect::<Vec<_>>();

                        self.update(module_id, &items, vis, ImportType::Glob);
                    } else {
                        // glob import from same crate => we do an initial
                        // import, and then need to propagate any further
                        // additions
                        let scope = &self.def_map[m.local_id].scope;

                        // Module scoped macros is included
                        let items = scope
                            .resolutions()
                            // only keep visible names...
                            .map(|(n, res)| {
                                (
                                    n,
                                    res.filter_visibility(|v| {
                                        v.is_visible_from_def_map(&self.def_map, module_id)
                                    }),
                                )
                            })
                            .filter(|(_, res)| !res.is_none())
                            .collect::<Vec<_>>();

                        self.update(module_id, &items, vis, ImportType::Glob);
                        // record the glob import in case we add further items
                        let glob = self.glob_imports.entry(m.local_id).or_default();
                        if !glob.iter().any(|(mid, _)| *mid == module_id) {
                            glob.push((module_id, vis));
                        }
                    }
                }
                Some(ModuleDefId::AdtId(AdtId::EnumId(e))) => {
                    mark::hit!(glob_enum);
                    // glob import from enum => just import all the variants

                    // XXX: urgh, so this works by accident! Here, we look at
                    // the enum data, and, in theory, this might require us to
                    // look back at the crate_def_map, creating a cycle. For
                    // example, `enum E { crate::some_macro!(); }`. Luckely, the
                    // only kind of macro that is allowed inside enum is a
                    // `cfg_macro`, and we don't need to run name resolution for
                    // it, but this is sheer luck!
                    let enum_data = self.db.enum_data(e);
                    let resolutions = enum_data
                        .variants
                        .iter()
                        .map(|(local_id, variant_data)| {
                            let name = variant_data.name.clone();
                            let variant = EnumVariantId { parent: e, local_id };
                            let res = PerNs::both(variant.into(), variant.into(), vis);
                            (Some(name), res)
                        })
                        .collect::<Vec<_>>();
                    self.update(module_id, &resolutions, vis, ImportType::Glob);
                }
                Some(d) => {
                    log::debug!("glob import {:?} from non-module/enum {:?}", import, d);
                }
                None => {
                    log::debug!("glob import {:?} didn't resolve as type", import);
                }
            }
        } else {
            match import.path.segments.last() {
                Some(last_segment) => {
                    let name = match &import.alias {
                        Some(ImportAlias::Alias(name)) => Some(name.clone()),
                        Some(ImportAlias::Underscore) => None,
                        None => Some(last_segment.clone()),
                    };
                    log::debug!("resolved import {:?} ({:?}) to {:?}", name, import, def);

                    // extern crates in the crate root are special-cased to insert entries into the extern prelude: rust-lang/rust#54658
                    if import.is_extern_crate && module_id == self.def_map.root {
                        if let (Some(def), Some(name)) = (def.take_types(), name.as_ref()) {
                            self.def_map.extern_prelude.insert(name.clone(), def);
                        }
                    }

                    self.update(module_id, &[(name, def)], vis, ImportType::Named);
                }
                None => mark::hit!(bogus_paths),
            }
        }
    }

    fn update(
        &mut self,
        module_id: LocalModuleId,
        resolutions: &[(Option<Name>, PerNs)],
        vis: Visibility,
        import_type: ImportType,
    ) {
        self.db.check_canceled();
        self.update_recursive(module_id, resolutions, vis, import_type, 0)
    }

    fn update_recursive(
        &mut self,
        module_id: LocalModuleId,
        resolutions: &[(Option<Name>, PerNs)],
        // All resolutions are imported with this visibility; the visibilies in
        // the `PerNs` values are ignored and overwritten
        vis: Visibility,
        import_type: ImportType,
        depth: usize,
    ) {
        if depth > GLOB_RECURSION_LIMIT {
            // prevent stack overflows (but this shouldn't be possible)
            panic!("infinite recursion in glob imports!");
        }
        let mut changed = false;

        for (name, res) in resolutions {
            match name {
                Some(name) => {
                    let scope = &mut self.def_map.modules[module_id].scope;
                    changed |= scope.push_res_with_import(
                        &mut self.from_glob_import,
                        (module_id, name.clone()),
                        res.with_visibility(vis),
                        import_type,
                    );
                }
                None => {
                    let tr = match res.take_types() {
                        Some(ModuleDefId::TraitId(tr)) => tr,
                        Some(other) => {
                            log::debug!("non-trait `_` import of {:?}", other);
                            continue;
                        }
                        None => continue,
                    };
                    let old_vis = self.def_map.modules[module_id].scope.unnamed_trait_vis(tr);
                    let should_update = match old_vis {
                        None => true,
                        Some(old_vis) => {
                            let max_vis = old_vis.max(vis, &self.def_map).unwrap_or_else(|| {
                                panic!("`Tr as _` imports with unrelated visibilities {:?} and {:?} (trait {:?})", old_vis, vis, tr);
                            });

                            if max_vis == old_vis {
                                false
                            } else {
                                mark::hit!(upgrade_underscore_visibility);
                                true
                            }
                        }
                    };

                    if should_update {
                        changed = true;
                        self.def_map.modules[module_id].scope.push_unnamed_trait(tr, vis);
                    }
                }
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
            .filter(|(glob_importing_module, _)| {
                // we know all resolutions have the same visibility (`vis`), so we
                // just need to check that once
                vis.is_visible_from_def_map(&self.def_map, *glob_importing_module)
            })
            .cloned()
            .collect::<Vec<_>>();

        for (glob_importing_module, glob_import_vis) in glob_imports {
            self.update_recursive(
                glob_importing_module,
                resolutions,
                glob_import_vis,
                ImportType::Glob,
                depth + 1,
            );
        }
    }

    fn resolve_macros(&mut self) -> ReachedFixedPoint {
        let mut macros = std::mem::replace(&mut self.unexpanded_macros, Vec::new());
        let mut attribute_macros =
            std::mem::replace(&mut self.unexpanded_attribute_macros, Vec::new());
        let mut resolved = Vec::new();
        let mut res = ReachedFixedPoint::Yes;
        macros.retain(|directive| {
            if let Some(call_id) = directive.legacy {
                res = ReachedFixedPoint::No;
                resolved.push((directive.module_id, call_id, directive.depth));
                return false;
            }

            if let Some(call_id) =
                directive.ast_id.as_call_id(self.db, self.def_map.krate, |path| {
                    let resolved_res = self.def_map.resolve_path_fp_with_macro(
                        self.db,
                        ResolveMode::Other,
                        directive.module_id,
                        &path,
                        BuiltinShadowMode::Module,
                    );
                    resolved_res.resolved_def.take_macros()
                })
            {
                resolved.push((directive.module_id, call_id, directive.depth));
                res = ReachedFixedPoint::No;
                return false;
            }

            true
        });
        attribute_macros.retain(|directive| {
            if let Some(call_id) =
                directive.ast_id.as_call_id(self.db, self.def_map.krate, |path| {
                    self.resolve_attribute_macro(&directive, &path)
                })
            {
                resolved.push((directive.module_id, call_id, 0));
                res = ReachedFixedPoint::No;
                return false;
            }

            true
        });

        self.unexpanded_macros = macros;
        self.unexpanded_attribute_macros = attribute_macros;

        for (module_id, macro_call_id, depth) in resolved {
            self.collect_macro_expansion(module_id, macro_call_id, depth);
        }

        res
    }

    fn resolve_attribute_macro(
        &self,
        directive: &DeriveDirective,
        path: &ModPath,
    ) -> Option<MacroDefId> {
        if let Some(name) = path.as_ident() {
            // FIXME this should actually be handled with the normal name
            // resolution; the std lib defines built-in stubs for the derives,
            // but these are new-style `macro`s, which we don't support yet
            if let Some(def_id) = find_builtin_derive(name) {
                return Some(def_id);
            }
        }
        let resolved_res = self.def_map.resolve_path_fp_with_macro(
            self.db,
            ResolveMode::Other,
            directive.module_id,
            &path,
            BuiltinShadowMode::Module,
        );

        resolved_res.resolved_def.take_macros()
    }

    fn collect_macro_expansion(
        &mut self,
        module_id: LocalModuleId,
        macro_call_id: MacroCallId,
        depth: usize,
    ) {
        if depth > EXPANSION_DEPTH_LIMIT {
            mark::hit!(macro_expansion_overflow);
            log::warn!("macro expansion is too deep");
            return;
        }
        let file_id: HirFileId = macro_call_id.as_file();
        let item_tree = self.db.item_tree(file_id);
        let mod_dir = self.mod_dirs[&module_id].clone();
        ModCollector {
            def_collector: &mut *self,
            macro_depth: depth,
            file_id,
            module_id,
            item_tree: &item_tree,
            mod_dir,
        }
        .collect(item_tree.top_level_items());
    }

    fn finish(self) -> CrateDefMap {
        self.def_map
    }
}

/// Walks a single module, populating defs, imports and macros
struct ModCollector<'a, 'b> {
    def_collector: &'a mut DefCollector<'b>,
    macro_depth: usize,
    module_id: LocalModuleId,
    file_id: HirFileId,
    item_tree: &'a ItemTree,
    mod_dir: ModDir,
}

impl ModCollector<'_, '_> {
    fn collect(&mut self, items: &[ModItem]) {
        // Note: don't assert that inserted value is fresh: it's simply not true
        // for macros.
        self.def_collector.mod_dirs.insert(self.module_id, self.mod_dir.clone());

        // Prelude module is always considered to be `#[macro_use]`.
        if let Some(prelude_module) = self.def_collector.def_map.prelude {
            if prelude_module.krate != self.def_collector.def_map.krate {
                mark::hit!(prelude_is_macro_use);
                self.def_collector.import_all_macros_exported(self.module_id, prelude_module.krate);
            }
        }

        // This should be processed eagerly instead of deferred to resolving.
        // `#[macro_use] extern crate` is hoisted to imports macros before collecting
        // any other items.
        for item in items {
            if self.is_cfg_enabled(self.item_tree.attrs((*item).into())) {
                if let ModItem::ExternCrate(id) = item {
                    let import = self.item_tree[*id].clone();
                    if import.is_macro_use {
                        self.def_collector.import_macros_from_extern_crate(self.module_id, &import);
                    }
                }
            }
        }

        for &item in items {
            let attrs = self.item_tree.attrs(item.into());
            if self.is_cfg_enabled(attrs) {
                let module =
                    ModuleId { krate: self.def_collector.def_map.krate, local_id: self.module_id };
                let container = ContainerId::ModuleId(module);

                let mut def = None;
                match item {
                    ModItem::Mod(m) => self.collect_module(&self.item_tree[m], attrs),
                    ModItem::Import(import_id) => {
                        self.def_collector.unresolved_imports.push(ImportDirective {
                            module_id: self.module_id,
                            import: Import::from_use(&self.item_tree, import_id),
                            status: PartialResolvedImport::Unresolved,
                        })
                    }
                    ModItem::ExternCrate(import_id) => {
                        self.def_collector.unresolved_imports.push(ImportDirective {
                            module_id: self.module_id,
                            import: Import::from_extern_crate(&self.item_tree, import_id),
                            status: PartialResolvedImport::Unresolved,
                        })
                    }
                    ModItem::MacroCall(mac) => self.collect_macro(&self.item_tree[mac]),
                    ModItem::Impl(imp) => {
                        let module = ModuleId {
                            krate: self.def_collector.def_map.krate,
                            local_id: self.module_id,
                        };
                        let container = ContainerId::ModuleId(module);
                        let impl_id = ImplLoc { container, id: ItemTreeId::new(self.file_id, imp) }
                            .intern(self.def_collector.db);
                        self.def_collector.def_map.modules[self.module_id]
                            .scope
                            .define_impl(impl_id)
                    }
                    ModItem::Function(id) => {
                        let func = &self.item_tree[id];
                        def = Some(DefData {
                            id: FunctionLoc {
                                container: container.into(),
                                id: ItemTreeId::new(self.file_id, id),
                            }
                            .intern(self.def_collector.db)
                            .into(),
                            name: &func.name,
                            visibility: &self.item_tree[func.visibility],
                            has_constructor: false,
                        });
                    }
                    ModItem::Struct(id) => {
                        let it = &self.item_tree[id];

                        // FIXME: check attrs to see if this is an attribute macro invocation;
                        // in which case we don't add the invocation, just a single attribute
                        // macro invocation
                        self.collect_derives(attrs, it.ast_id.upcast());

                        def = Some(DefData {
                            id: StructLoc { container, id: ItemTreeId::new(self.file_id, id) }
                                .intern(self.def_collector.db)
                                .into(),
                            name: &it.name,
                            visibility: &self.item_tree[it.visibility],
                            has_constructor: it.kind != StructDefKind::Record,
                        });
                    }
                    ModItem::Union(id) => {
                        let it = &self.item_tree[id];

                        // FIXME: check attrs to see if this is an attribute macro invocation;
                        // in which case we don't add the invocation, just a single attribute
                        // macro invocation
                        self.collect_derives(attrs, it.ast_id.upcast());

                        def = Some(DefData {
                            id: UnionLoc { container, id: ItemTreeId::new(self.file_id, id) }
                                .intern(self.def_collector.db)
                                .into(),
                            name: &it.name,
                            visibility: &self.item_tree[it.visibility],
                            has_constructor: false,
                        });
                    }
                    ModItem::Enum(id) => {
                        let it = &self.item_tree[id];

                        // FIXME: check attrs to see if this is an attribute macro invocation;
                        // in which case we don't add the invocation, just a single attribute
                        // macro invocation
                        self.collect_derives(attrs, it.ast_id.upcast());

                        def = Some(DefData {
                            id: EnumLoc { container, id: ItemTreeId::new(self.file_id, id) }
                                .intern(self.def_collector.db)
                                .into(),
                            name: &it.name,
                            visibility: &self.item_tree[it.visibility],
                            has_constructor: false,
                        });
                    }
                    ModItem::Const(id) => {
                        let it = &self.item_tree[id];

                        if let Some(name) = &it.name {
                            def = Some(DefData {
                                id: ConstLoc {
                                    container: container.into(),
                                    id: ItemTreeId::new(self.file_id, id),
                                }
                                .intern(self.def_collector.db)
                                .into(),
                                name,
                                visibility: &self.item_tree[it.visibility],
                                has_constructor: false,
                            });
                        }
                    }
                    ModItem::Static(id) => {
                        let it = &self.item_tree[id];

                        def = Some(DefData {
                            id: StaticLoc { container, id: ItemTreeId::new(self.file_id, id) }
                                .intern(self.def_collector.db)
                                .into(),
                            name: &it.name,
                            visibility: &self.item_tree[it.visibility],
                            has_constructor: false,
                        });
                    }
                    ModItem::Trait(id) => {
                        let it = &self.item_tree[id];

                        def = Some(DefData {
                            id: TraitLoc { container, id: ItemTreeId::new(self.file_id, id) }
                                .intern(self.def_collector.db)
                                .into(),
                            name: &it.name,
                            visibility: &self.item_tree[it.visibility],
                            has_constructor: false,
                        });
                    }
                    ModItem::TypeAlias(id) => {
                        let it = &self.item_tree[id];

                        def = Some(DefData {
                            id: TypeAliasLoc {
                                container: container.into(),
                                id: ItemTreeId::new(self.file_id, id),
                            }
                            .intern(self.def_collector.db)
                            .into(),
                            name: &it.name,
                            visibility: &self.item_tree[it.visibility],
                            has_constructor: false,
                        });
                    }
                }

                if let Some(DefData { id, name, visibility, has_constructor }) = def {
                    self.def_collector.def_map.modules[self.module_id].scope.define_def(id);
                    let vis = self
                        .def_collector
                        .def_map
                        .resolve_visibility(self.def_collector.db, self.module_id, visibility)
                        .unwrap_or(Visibility::Public);
                    self.def_collector.update(
                        self.module_id,
                        &[(Some(name.clone()), PerNs::from_def(id, vis, has_constructor))],
                        vis,
                        ImportType::Named,
                    )
                }
            }
        }
    }

    fn collect_module(&mut self, module: &Mod, attrs: &Attrs) {
        let path_attr = attrs.by_key("path").string_value();
        let is_macro_use = attrs.by_key("macro_use").exists();
        match &module.kind {
            // inline module, just recurse
            ModKind::Inline { items } => {
                let module_id = self.push_child_module(
                    module.name.clone(),
                    AstId::new(self.file_id, module.ast_id),
                    None,
                    &self.item_tree[module.visibility],
                );

                ModCollector {
                    def_collector: &mut *self.def_collector,
                    macro_depth: self.macro_depth,
                    module_id,
                    file_id: self.file_id,
                    item_tree: self.item_tree,
                    mod_dir: self.mod_dir.descend_into_definition(&module.name, path_attr),
                }
                .collect(&*items);
                if is_macro_use {
                    self.import_all_legacy_macros(module_id);
                }
            }
            // out of line module, resolve, parse and recurse
            ModKind::Outline {} => {
                let ast_id = AstId::new(self.file_id, module.ast_id);
                match self.mod_dir.resolve_declaration(
                    self.def_collector.db,
                    self.file_id,
                    &module.name,
                    path_attr,
                ) {
                    Ok((file_id, is_mod_rs, mod_dir)) => {
                        let module_id = self.push_child_module(
                            module.name.clone(),
                            ast_id,
                            Some((file_id, is_mod_rs)),
                            &self.item_tree[module.visibility],
                        );
                        let item_tree = self.def_collector.db.item_tree(file_id.into());
                        ModCollector {
                            def_collector: &mut *self.def_collector,
                            macro_depth: self.macro_depth,
                            module_id,
                            file_id: file_id.into(),
                            item_tree: &item_tree,
                            mod_dir,
                        }
                        .collect(item_tree.top_level_items());
                        if is_macro_use {
                            self.import_all_legacy_macros(module_id);
                        }
                    }
                    Err(candidate) => self.def_collector.def_map.diagnostics.push(
                        DefDiagnostic::UnresolvedModule {
                            module: self.module_id,
                            declaration: ast_id,
                            candidate,
                        },
                    ),
                };
            }
        }
    }

    fn push_child_module(
        &mut self,
        name: Name,
        declaration: AstId<ast::Module>,
        definition: Option<(FileId, bool)>,
        visibility: &crate::visibility::RawVisibility,
    ) -> LocalModuleId {
        let vis = self
            .def_collector
            .def_map
            .resolve_visibility(self.def_collector.db, self.module_id, visibility)
            .unwrap_or(Visibility::Public);
        let modules = &mut self.def_collector.def_map.modules;
        let res = modules.alloc(ModuleData::default());
        modules[res].parent = Some(self.module_id);
        modules[res].origin = match definition {
            None => ModuleOrigin::Inline { definition: declaration },
            Some((definition, is_mod_rs)) => {
                ModuleOrigin::File { declaration, definition, is_mod_rs }
            }
        };
        for (name, mac) in modules[self.module_id].scope.collect_legacy_macros() {
            modules[res].scope.define_legacy_macro(name, mac)
        }
        modules[self.module_id].children.insert(name.clone(), res);
        let module = ModuleId { krate: self.def_collector.def_map.krate, local_id: res };
        let def: ModuleDefId = module.into();
        self.def_collector.def_map.modules[self.module_id].scope.define_def(def);
        self.def_collector.update(
            self.module_id,
            &[(Some(name), PerNs::from_def(def, vis, false))],
            vis,
            ImportType::Named,
        );
        res
    }

    fn collect_derives(&mut self, attrs: &Attrs, ast_id: FileAstId<ast::Item>) {
        for derive_subtree in attrs.by_key("derive").tt_values() {
            // for #[derive(Copy, Clone)], `derive_subtree` is the `(Copy, Clone)` subtree
            for tt in &derive_subtree.token_trees {
                let ident = match &tt {
                    tt::TokenTree::Leaf(tt::Leaf::Ident(ident)) => ident,
                    tt::TokenTree::Leaf(tt::Leaf::Punct(_)) => continue, // , is ok
                    _ => continue, // anything else would be an error (which we currently ignore)
                };
                let path = ModPath::from_tt_ident(ident);

                let ast_id = AstIdWithPath::new(self.file_id, ast_id, path);
                self.def_collector
                    .unexpanded_attribute_macros
                    .push(DeriveDirective { module_id: self.module_id, ast_id });
            }
        }
    }

    fn collect_macro(&mut self, mac: &MacroCall) {
        let mut ast_id = AstIdWithPath::new(self.file_id, mac.ast_id, mac.path.clone());

        // Case 0: builtin macros
        if mac.is_builtin {
            if let Some(name) = &mac.name {
                let krate = self.def_collector.def_map.krate;
                if let Some(macro_id) = find_builtin_macro(name, krate, ast_id.ast_id) {
                    self.def_collector.define_macro(
                        self.module_id,
                        name.clone(),
                        macro_id,
                        mac.is_export,
                    );
                    return;
                }
            }
        }

        // Case 1: macro rules, define a macro in crate-global mutable scope
        if is_macro_rules(&mac.path) {
            if let Some(name) = &mac.name {
                let macro_id = MacroDefId {
                    ast_id: Some(ast_id.ast_id),
                    krate: Some(self.def_collector.def_map.krate),
                    kind: MacroDefKind::Declarative,
                    local_inner: mac.is_local_inner,
                };
                self.def_collector.define_macro(
                    self.module_id,
                    name.clone(),
                    macro_id,
                    mac.is_export,
                );
            }
            return;
        }

        // Case 2: try to resolve in legacy scope and expand macro_rules
        if let Some(macro_call_id) =
            ast_id.as_call_id(self.def_collector.db, self.def_collector.def_map.krate, |path| {
                path.as_ident().and_then(|name| {
                    self.def_collector.def_map[self.module_id].scope.get_legacy_macro(&name)
                })
            })
        {
            self.def_collector.unexpanded_macros.push(MacroDirective {
                module_id: self.module_id,
                ast_id,
                legacy: Some(macro_call_id),
                depth: self.macro_depth + 1,
            });

            return;
        }

        // Case 3: resolve in module scope, expand during name resolution.
        // We rewrite simple path `macro_name` to `self::macro_name` to force resolve in module scope only.
        if ast_id.path.is_ident() {
            ast_id.path.kind = PathKind::Super(0);
        }

        self.def_collector.unexpanded_macros.push(MacroDirective {
            module_id: self.module_id,
            ast_id,
            legacy: None,
            depth: self.macro_depth + 1,
        });
    }

    fn import_all_legacy_macros(&mut self, module_id: LocalModuleId) {
        let macros = self.def_collector.def_map[module_id].scope.collect_legacy_macros();
        for (name, macro_) in macros {
            self.def_collector.define_legacy_macro(self.module_id, name.clone(), macro_);
        }
    }

    fn is_cfg_enabled(&self, attrs: &Attrs) -> bool {
        attrs.is_cfg_enabled(self.def_collector.cfg_options)
    }
}

fn is_macro_rules(path: &ModPath) -> bool {
    path.as_ident() == Some(&name![macro_rules])
}

#[cfg(test)]
mod tests {
    use crate::{db::DefDatabase, test_db::TestDB};
    use arena::Arena;
    use base_db::{fixture::WithFixture, SourceDatabase};

    use super::*;

    fn do_collect_defs(db: &dyn DefDatabase, def_map: CrateDefMap) -> CrateDefMap {
        let mut collector = DefCollector {
            db,
            def_map,
            glob_imports: FxHashMap::default(),
            unresolved_imports: Vec::new(),
            resolved_imports: Vec::new(),
            unexpanded_macros: Vec::new(),
            unexpanded_attribute_macros: Vec::new(),
            mod_dirs: FxHashMap::default(),
            cfg_options: &CfgOptions::default(),
            proc_macros: Default::default(),
            from_glob_import: Default::default(),
        };
        collector.collect();
        collector.def_map
    }

    fn do_resolve(code: &str) -> CrateDefMap {
        let (db, _file_id) = TestDB::with_single_file(&code);
        let krate = db.test_crate();

        let def_map = {
            let edition = db.crate_graph()[krate].edition;
            let mut modules: Arena<ModuleData> = Arena::default();
            let root = modules.alloc(ModuleData::default());
            CrateDefMap {
                krate,
                edition,
                extern_prelude: FxHashMap::default(),
                prelude: None,
                root,
                modules,
                diagnostics: Vec::new(),
            }
        };
        do_collect_defs(&db, def_map)
    }

    #[test]
    fn test_macro_expand_will_stop_1() {
        do_resolve(
            r#"
        macro_rules! foo {
            ($($ty:ty)*) => { foo!($($ty)*); }
        }
        foo!(KABOOM);
        "#,
        );
    }

    #[ignore] // this test does succeed, but takes quite a while :/
    #[test]
    fn test_macro_expand_will_stop_2() {
        do_resolve(
            r#"
        macro_rules! foo {
            ($($ty:ty)*) => { foo!($($ty)* $($ty)*); }
        }
        foo!(KABOOM);
        "#,
        );
    }
}
