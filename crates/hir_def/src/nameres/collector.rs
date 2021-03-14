//! The core of the module-level name resolution algorithm.
//!
//! `DefCollector::collect` contains the fixed-point iteration loop which
//! resolves imports and expands macros.

use std::iter;

use base_db::{CrateId, FileId, ProcMacroId};
use cfg::{CfgExpr, CfgOptions};
use hir_expand::{
    ast_id_map::FileAstId,
    builtin_derive::find_builtin_derive,
    builtin_macro::find_builtin_macro,
    name::{AsName, Name},
    proc_macro::ProcMacroExpander,
    HirFileId, MacroCallId, MacroCallKind, MacroDefId, MacroDefKind,
};
use hir_expand::{InFile, MacroCallLoc};
use rustc_hash::{FxHashMap, FxHashSet};
use syntax::ast;
use tt::{Leaf, TokenTree};

use crate::{
    attr::Attrs,
    db::DefDatabase,
    item_attr_as_call_id,
    item_scope::{ImportType, PerNsGlobImports},
    item_tree::{
        self, FileItemTreeId, ItemTree, ItemTreeId, MacroCall, MacroRules, Mod, ModItem, ModKind,
        StructDefKind,
    },
    macro_call_as_call_id,
    nameres::{
        diagnostics::DefDiagnostic, mod_resolution::ModDir, path_resolution::ReachedFixedPoint,
        BuiltinShadowMode, DefMap, ModuleData, ModuleOrigin, ResolveMode,
    },
    path::{ImportAlias, ModPath, PathKind},
    per_ns::PerNs,
    visibility::{RawVisibility, Visibility},
    AdtId, AstId, AstIdWithPath, ConstLoc, EnumLoc, EnumVariantId, FunctionLoc, ImplLoc, Intern,
    LocalModuleId, ModuleDefId, StaticLoc, StructLoc, TraitLoc, TypeAliasLoc, UnionLoc,
    UnresolvedMacro,
};

const GLOB_RECURSION_LIMIT: usize = 100;
const EXPANSION_DEPTH_LIMIT: usize = 128;
const FIXED_POINT_LIMIT: usize = 8192;

pub(super) fn collect_defs(
    db: &dyn DefDatabase,
    mut def_map: DefMap,
    block: Option<AstId<ast::BlockExpr>>,
) -> DefMap {
    let crate_graph = db.crate_graph();

    // populate external prelude
    for dep in &crate_graph[def_map.krate].dependencies {
        log::debug!("crate dep {:?} -> {:?}", dep.name, dep.crate_id);
        let dep_def_map = db.crate_def_map(dep.crate_id);
        def_map
            .extern_prelude
            .insert(dep.as_name(), dep_def_map.module_id(dep_def_map.root).into());

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
        exports_proc_macros: false,
        from_glob_import: Default::default(),
    };
    match block {
        Some(block) => {
            collector.seed_with_inner(block);
        }
        None => {
            collector.seed_with_top_level();
        }
    }
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
enum ImportSource {
    Import(ItemTreeId<item_tree::Import>),
    ExternCrate(ItemTreeId<item_tree::ExternCrate>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Import {
    path: ModPath,
    alias: Option<ImportAlias>,
    visibility: RawVisibility,
    is_glob: bool,
    is_prelude: bool,
    is_extern_crate: bool,
    is_macro_use: bool,
    source: ImportSource,
}

impl Import {
    fn from_use(
        db: &dyn DefDatabase,
        krate: CrateId,
        tree: &ItemTree,
        id: ItemTreeId<item_tree::Import>,
    ) -> Self {
        let it = &tree[id.value];
        let attrs = &tree.attrs(db, krate, ModItem::from(id.value).into());
        let visibility = &tree[it.visibility];
        Self {
            path: it.path.clone(),
            alias: it.alias.clone(),
            visibility: visibility.clone(),
            is_glob: it.is_glob,
            is_prelude: attrs.by_key("prelude_import").exists(),
            is_extern_crate: false,
            is_macro_use: false,
            source: ImportSource::Import(id),
        }
    }

    fn from_extern_crate(
        db: &dyn DefDatabase,
        krate: CrateId,
        tree: &ItemTree,
        id: ItemTreeId<item_tree::ExternCrate>,
    ) -> Self {
        let it = &tree[id.value];
        let attrs = &tree.attrs(db, krate, ModItem::from(id.value).into());
        let visibility = &tree[it.visibility];
        Self {
            path: ModPath::from_segments(PathKind::Plain, iter::once(it.name.clone())),
            alias: it.alias.clone(),
            visibility: visibility.clone(),
            is_glob: false,
            is_prelude: false,
            is_extern_crate: true,
            is_macro_use: attrs.by_key("macro_use").exists(),
            source: ImportSource::ExternCrate(id),
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
    def_map: DefMap,
    glob_imports: FxHashMap<LocalModuleId, Vec<(LocalModuleId, Visibility)>>,
    unresolved_imports: Vec<ImportDirective>,
    resolved_imports: Vec<ImportDirective>,
    unexpanded_macros: Vec<MacroDirective>,
    unexpanded_attribute_macros: Vec<DeriveDirective>,
    mod_dirs: FxHashMap<LocalModuleId, ModDir>,
    cfg_options: &'a CfgOptions,
    /// List of procedural macros defined by this crate. This is read from the dynamic library
    /// built by the build system, and is the list of proc. macros we can actually expand. It is
    /// empty when proc. macro support is disabled (in which case we still do name resolution for
    /// them).
    proc_macros: Vec<(Name, ProcMacroExpander)>,
    exports_proc_macros: bool,
    from_glob_import: PerNsGlobImports,
}

impl DefCollector<'_> {
    fn seed_with_top_level(&mut self) {
        let file_id = self.db.crate_graph()[self.def_map.krate].root_file_id;
        let item_tree = self.db.item_tree(file_id.into());
        let module_id = self.def_map.root;
        self.def_map.modules[module_id].origin = ModuleOrigin::CrateRoot { definition: file_id };
        if item_tree
            .top_level_attrs(self.db, self.def_map.krate)
            .cfg()
            .map_or(true, |cfg| self.cfg_options.check(&cfg) != Some(false))
        {
            ModCollector {
                def_collector: &mut *self,
                macro_depth: 0,
                module_id,
                file_id: file_id.into(),
                item_tree: &item_tree,
                mod_dir: ModDir::root(),
            }
            .collect(item_tree.top_level_items());
        }
    }

    fn seed_with_inner(&mut self, block: AstId<ast::BlockExpr>) {
        let item_tree = self.db.item_tree(block.file_id);
        let module_id = self.def_map.root;
        self.def_map.modules[module_id].origin = ModuleOrigin::BlockExpr { block };
        if item_tree
            .top_level_attrs(self.db, self.def_map.krate)
            .cfg()
            .map_or(true, |cfg| self.cfg_options.check(&cfg) != Some(false))
        {
            ModCollector {
                def_collector: &mut *self,
                macro_depth: 0,
                module_id,
                file_id: block.file_id,
                item_tree: &item_tree,
                mod_dir: ModDir::root(),
            }
            .collect(item_tree.inner_items_of_block(block.value));
        }
    }

    fn collect(&mut self) {
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
        // FIXME: We maybe could skip this, if we handle the indeterminate imports in `resolve_imports`
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
        for directive in &unresolved_imports {
            self.record_resolved_import(directive)
        }
        self.unresolved_imports = unresolved_imports;

        // FIXME: This condition should instead check if this is a `proc-macro` type crate.
        if self.exports_proc_macros {
            // A crate exporting procedural macros is not allowed to export anything else.
            //
            // Additionally, while the proc macro entry points must be `pub`, they are not publicly
            // exported in type/value namespace. This function reduces the visibility of all items
            // in the crate root that aren't proc macros.
            let root = self.def_map.root;
            let module_id = self.def_map.module_id(root);
            let root = &mut self.def_map.modules[root];
            root.scope.censor_non_proc_macros(module_id);
        }
    }

    /// Adds a definition of procedural macro `name` to the root module.
    ///
    /// # Notes on procedural macro resolution
    ///
    /// Procedural macro functionality is provided by the build system: It has to build the proc
    /// macro and pass the resulting dynamic library to rust-analyzer.
    ///
    /// When procedural macro support is enabled, the list of proc macros exported by a crate is
    /// known before we resolve names in the crate. This list is stored in `self.proc_macros` and is
    /// derived from the dynamic library.
    ///
    /// However, we *also* would like to be able to at least *resolve* macros on our own, without
    /// help by the build system. So, when the macro isn't found in `self.proc_macros`, we instead
    /// use a dummy expander that always errors. This comes with the drawback of macros potentially
    /// going out of sync with what the build system sees (since we resolve using VFS state, but
    /// Cargo builds only on-disk files). We could and probably should add diagnostics for that.
    fn resolve_proc_macro(&mut self, name: &Name) {
        self.exports_proc_macros = true;
        let macro_def = match self.proc_macros.iter().find(|(n, _)| n == name) {
            Some((_, expander)) => MacroDefId {
                ast_id: None,
                krate: self.def_map.krate,
                kind: MacroDefKind::ProcMacro(*expander),
                local_inner: false,
            },
            None => MacroDefId {
                ast_id: None,
                krate: self.def_map.krate,
                kind: MacroDefKind::ProcMacro(ProcMacroExpander::dummy(self.def_map.krate)),
                local_inner: false,
            },
        };

        self.define_proc_macro(name.clone(), macro_def);
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
    /// A proc macro is similar to normal macro scope, but it would not visible in legacy textual scoped.
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
        extern_crate: &item_tree::ExternCrate,
    ) {
        log::debug!(
            "importing macros from extern crate: {:?} ({:?})",
            extern_crate,
            self.def_map.edition,
        );

        let res = self.def_map.resolve_name_in_extern_prelude(&extern_crate.name);

        if let Some(ModuleDefId::ModuleId(m)) = res.take_types() {
            cov_mark::hit!(macro_rules_from_other_crates_are_visible_with_macro_use);
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
            if res.is_none() {
                PartialResolvedImport::Unresolved
            } else {
                PartialResolvedImport::Resolved(res)
            }
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
                        cov_mark::hit!(std_prelude);
                        self.def_map.prelude = Some(m);
                    } else if m.krate != self.def_map.krate {
                        cov_mark::hit!(glob_across_crates);
                        // glob import from other crate => we can just import everything once
                        let item_map = m.def_map(self.db);
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
                        let def_map;
                        let scope = if m.block == self.def_map.block_id() {
                            &self.def_map[m.local_id].scope
                        } else {
                            def_map = m.def_map(self.db);
                            &def_map[m.local_id].scope
                        };

                        // Module scoped macros is included
                        let items = scope
                            .resolutions()
                            // only keep visible names...
                            .map(|(n, res)| {
                                (
                                    n,
                                    res.filter_visibility(|v| {
                                        v.is_visible_from_def_map(self.db, &self.def_map, module_id)
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
                    cov_mark::hit!(glob_enum);
                    // glob import from enum => just import all the variants

                    // XXX: urgh, so this works by accident! Here, we look at
                    // the enum data, and, in theory, this might require us to
                    // look back at the crate_def_map, creating a cycle. For
                    // example, `enum E { crate::some_macro!(); }`. Luckily, the
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
            match import.path.segments().last() {
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
                None => cov_mark::hit!(bogus_paths),
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
        // All resolutions are imported with this visibility; the visibilities in
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
                                cov_mark::hit!(upgrade_underscore_visibility);
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
                vis.is_visible_from_def_map(self.db, &self.def_map, *glob_importing_module)
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

            match macro_call_as_call_id(
                &directive.ast_id,
                self.db,
                self.def_map.krate,
                |path| {
                    let resolved_res = self.def_map.resolve_path_fp_with_macro(
                        self.db,
                        ResolveMode::Other,
                        directive.module_id,
                        &path,
                        BuiltinShadowMode::Module,
                    );
                    resolved_res.resolved_def.take_macros()
                },
                &mut |_err| (),
            ) {
                Ok(Ok(call_id)) => {
                    resolved.push((directive.module_id, call_id, directive.depth));
                    res = ReachedFixedPoint::No;
                    return false;
                }
                Err(UnresolvedMacro) | Ok(Err(_)) => {}
            }

            true
        });
        attribute_macros.retain(|directive| {
            match item_attr_as_call_id(&directive.ast_id, self.db, self.def_map.krate, |path| {
                self.resolve_attribute_macro(&directive, &path)
            }) {
                Ok(call_id) => {
                    resolved.push((directive.module_id, call_id, 0));
                    res = ReachedFixedPoint::No;
                    return false;
                }
                Err(UnresolvedMacro) => (),
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
            cov_mark::hit!(macro_expansion_overflow);
            log::warn!("macro expansion is too deep");
            return;
        }
        let file_id = macro_call_id.as_file();

        // First, fetch the raw expansion result for purposes of error reporting. This goes through
        // `macro_expand_error` to avoid depending on the full expansion result (to improve
        // incrementality).
        let err = self.db.macro_expand_error(macro_call_id);
        if let Some(err) = err {
            if let MacroCallId::LazyMacro(id) = macro_call_id {
                let loc: MacroCallLoc = self.db.lookup_intern_macro(id);

                let diag = match err {
                    hir_expand::ExpandError::UnresolvedProcMacro => {
                        // Missing proc macros are non-fatal, so they are handled specially.
                        DefDiagnostic::unresolved_proc_macro(module_id, loc.kind)
                    }
                    _ => DefDiagnostic::macro_error(module_id, loc.kind, err.to_string()),
                };

                self.def_map.diagnostics.push(diag);
            }
            // FIXME: Handle eager macros.
        }

        // Then, fetch and process the item tree. This will reuse the expansion result from above.
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

    fn finish(mut self) -> DefMap {
        // Emit diagnostics for all remaining unexpanded macros.

        for directive in &self.unexpanded_macros {
            let mut error = None;
            match macro_call_as_call_id(
                &directive.ast_id,
                self.db,
                self.def_map.krate,
                |path| {
                    let resolved_res = self.def_map.resolve_path_fp_with_macro(
                        self.db,
                        ResolveMode::Other,
                        directive.module_id,
                        &path,
                        BuiltinShadowMode::Module,
                    );
                    resolved_res.resolved_def.take_macros()
                },
                &mut |e| {
                    error.get_or_insert(e);
                },
            ) {
                Ok(_) => (),
                Err(UnresolvedMacro) => {
                    self.def_map.diagnostics.push(DefDiagnostic::unresolved_macro_call(
                        directive.module_id,
                        directive.ast_id.ast_id,
                    ));
                }
            };
        }

        // Emit diagnostics for all remaining unresolved imports.

        // We'd like to avoid emitting a diagnostics avalanche when some `extern crate` doesn't
        // resolve. We first emit diagnostics for unresolved extern crates and collect the missing
        // crate names. Then we emit diagnostics for unresolved imports, but only if the import
        // doesn't start with an unresolved crate's name. Due to renaming and reexports, this is a
        // heuristic, but it works in practice.
        let mut diagnosed_extern_crates = FxHashSet::default();
        for directive in &self.unresolved_imports {
            if let ImportSource::ExternCrate(krate) = directive.import.source {
                let item_tree = self.db.item_tree(krate.file_id);
                let extern_crate = &item_tree[krate.value];

                diagnosed_extern_crates.insert(extern_crate.name.clone());

                self.def_map.diagnostics.push(DefDiagnostic::unresolved_extern_crate(
                    directive.module_id,
                    InFile::new(krate.file_id, extern_crate.ast_id),
                ));
            }
        }

        for directive in &self.unresolved_imports {
            if let ImportSource::Import(import) = &directive.import.source {
                let item_tree = self.db.item_tree(import.file_id);
                let import_data = &item_tree[import.value];

                match (import_data.path.segments().first(), &import_data.path.kind) {
                    (Some(krate), PathKind::Plain) | (Some(krate), PathKind::Abs) => {
                        if diagnosed_extern_crates.contains(krate) {
                            continue;
                        }
                    }
                    _ => {}
                }

                self.def_map.diagnostics.push(DefDiagnostic::unresolved_import(
                    directive.module_id,
                    InFile::new(import.file_id, import_data.ast_id),
                    import_data.index,
                ));
            }
        }

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
        let krate = self.def_collector.def_map.krate;

        // Note: don't assert that inserted value is fresh: it's simply not true
        // for macros.
        self.def_collector.mod_dirs.insert(self.module_id, self.mod_dir.clone());

        // Prelude module is always considered to be `#[macro_use]`.
        if let Some(prelude_module) = self.def_collector.def_map.prelude {
            if prelude_module.krate != self.def_collector.def_map.krate {
                cov_mark::hit!(prelude_is_macro_use);
                self.def_collector.import_all_macros_exported(self.module_id, prelude_module.krate);
            }
        }

        // This should be processed eagerly instead of deferred to resolving.
        // `#[macro_use] extern crate` is hoisted to imports macros before collecting
        // any other items.
        for item in items {
            let attrs = self.item_tree.attrs(self.def_collector.db, krate, (*item).into());
            if attrs.cfg().map_or(true, |cfg| self.is_cfg_enabled(&cfg)) {
                if let ModItem::ExternCrate(id) = item {
                    let import = self.item_tree[*id].clone();
                    let attrs = self.item_tree.attrs(
                        self.def_collector.db,
                        krate,
                        ModItem::from(*id).into(),
                    );
                    if attrs.by_key("macro_use").exists() {
                        self.def_collector.import_macros_from_extern_crate(self.module_id, &import);
                    }
                }
            }
        }

        for &item in items {
            let attrs = self.item_tree.attrs(self.def_collector.db, krate, item.into());
            if let Some(cfg) = attrs.cfg() {
                if !self.is_cfg_enabled(&cfg) {
                    self.emit_unconfigured_diagnostic(item, &cfg);
                    continue;
                }
            }
            let module = self.def_collector.def_map.module_id(self.module_id);

            let mut def = None;
            match item {
                ModItem::Mod(m) => self.collect_module(&self.item_tree[m], &attrs),
                ModItem::Import(import_id) => {
                    self.def_collector.unresolved_imports.push(ImportDirective {
                        module_id: self.module_id,
                        import: Import::from_use(
                            self.def_collector.db,
                            krate,
                            &self.item_tree,
                            InFile::new(self.file_id, import_id),
                        ),
                        status: PartialResolvedImport::Unresolved,
                    })
                }
                ModItem::ExternCrate(import_id) => {
                    self.def_collector.unresolved_imports.push(ImportDirective {
                        module_id: self.module_id,
                        import: Import::from_extern_crate(
                            self.def_collector.db,
                            krate,
                            &self.item_tree,
                            InFile::new(self.file_id, import_id),
                        ),
                        status: PartialResolvedImport::Unresolved,
                    })
                }
                ModItem::MacroCall(mac) => self.collect_macro_call(&self.item_tree[mac]),
                ModItem::MacroRules(id) => self.collect_macro_rules(id),
                ModItem::MacroDef(id) => {
                    let mac = &self.item_tree[id];
                    let ast_id = InFile::new(self.file_id, mac.ast_id.upcast());

                    // "Macro 2.0" is not currently supported by rust-analyzer, but libcore uses it
                    // to define builtin macros, so we support at least that part.
                    let attrs = self.item_tree.attrs(
                        self.def_collector.db,
                        krate,
                        ModItem::from(id).into(),
                    );
                    if attrs.by_key("rustc_builtin_macro").exists() {
                        let krate = self.def_collector.def_map.krate;
                        let macro_id = find_builtin_macro(&mac.name, krate, ast_id)
                            .or_else(|| find_builtin_derive(&mac.name, krate, ast_id));
                        if let Some(macro_id) = macro_id {
                            let vis = self
                                .def_collector
                                .def_map
                                .resolve_visibility(
                                    self.def_collector.db,
                                    self.module_id,
                                    &self.item_tree[mac.visibility],
                                )
                                .unwrap_or(Visibility::Public);
                            self.def_collector.update(
                                self.module_id,
                                &[(Some(mac.name.clone()), PerNs::macros(macro_id, vis))],
                                vis,
                                ImportType::Named,
                            );
                        }
                    }
                }
                ModItem::Impl(imp) => {
                    let module = self.def_collector.def_map.module_id(self.module_id);
                    let impl_id =
                        ImplLoc { container: module, id: ItemTreeId::new(self.file_id, imp) }
                            .intern(self.def_collector.db);
                    self.def_collector.def_map.modules[self.module_id].scope.define_impl(impl_id)
                }
                ModItem::Function(id) => {
                    let func = &self.item_tree[id];

                    self.collect_proc_macro_def(&func.name, &attrs);

                    def = Some(DefData {
                        id: FunctionLoc {
                            container: module.into(),
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
                    self.collect_derives(&attrs, it.ast_id.upcast());

                    def = Some(DefData {
                        id: StructLoc { container: module, id: ItemTreeId::new(self.file_id, id) }
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
                    self.collect_derives(&attrs, it.ast_id.upcast());

                    def = Some(DefData {
                        id: UnionLoc { container: module, id: ItemTreeId::new(self.file_id, id) }
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
                    self.collect_derives(&attrs, it.ast_id.upcast());

                    def = Some(DefData {
                        id: EnumLoc { container: module, id: ItemTreeId::new(self.file_id, id) }
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
                                container: module.into(),
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
                        id: StaticLoc { container: module, id: ItemTreeId::new(self.file_id, id) }
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
                        id: TraitLoc { container: module, id: ItemTreeId::new(self.file_id, id) }
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
                            container: module.into(),
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

                if let Some(mod_dir) = self.mod_dir.descend_into_definition(&module.name, path_attr)
                {
                    ModCollector {
                        def_collector: &mut *self.def_collector,
                        macro_depth: self.macro_depth,
                        module_id,
                        file_id: self.file_id,
                        item_tree: self.item_tree,
                        mod_dir,
                    }
                    .collect(&*items);
                    if is_macro_use {
                        self.import_all_legacy_macros(module_id);
                    }
                }
            }
            // out of line module, resolve, parse and recurse
            ModKind::Outline {} => {
                let ast_id = AstId::new(self.file_id, module.ast_id);
                let db = self.def_collector.db;
                match self.mod_dir.resolve_declaration(db, self.file_id, &module.name, path_attr) {
                    Ok((file_id, is_mod_rs, mod_dir)) => {
                        let module_id = self.push_child_module(
                            module.name.clone(),
                            ast_id,
                            Some((file_id, is_mod_rs)),
                            &self.item_tree[module.visibility],
                        );
                        let item_tree = db.item_tree(file_id.into());
                        ModCollector {
                            def_collector: &mut *self.def_collector,
                            macro_depth: self.macro_depth,
                            module_id,
                            file_id: file_id.into(),
                            item_tree: &item_tree,
                            mod_dir,
                        }
                        .collect(item_tree.top_level_items());
                        if is_macro_use
                            || item_tree
                                .top_level_attrs(db, self.def_collector.def_map.krate)
                                .by_key("macro_use")
                                .exists()
                        {
                            self.import_all_legacy_macros(module_id);
                        }
                    }
                    Err(candidate) => {
                        self.def_collector.def_map.diagnostics.push(
                            DefDiagnostic::unresolved_module(self.module_id, ast_id, candidate),
                        );
                    }
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
        let module = self.def_collector.def_map.module_id(res);
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
        for derive in attrs.by_key("derive").attrs() {
            match derive.parse_derive() {
                Some(derive_macros) => {
                    for path in derive_macros {
                        let ast_id = AstIdWithPath::new(self.file_id, ast_id, path);
                        self.def_collector
                            .unexpanded_attribute_macros
                            .push(DeriveDirective { module_id: self.module_id, ast_id });
                    }
                }
                None => {
                    // FIXME: diagnose
                    log::debug!("malformed derive: {:?}", derive);
                }
            }
        }
    }

    /// If `attrs` registers a procedural macro, collects its definition.
    fn collect_proc_macro_def(&mut self, func_name: &Name, attrs: &Attrs) {
        // FIXME: this should only be done in the root module of `proc-macro` crates, not everywhere
        // FIXME: distinguish the type of macro
        let macro_name = if attrs.by_key("proc_macro").exists()
            || attrs.by_key("proc_macro_attribute").exists()
        {
            func_name.clone()
        } else {
            let derive = attrs.by_key("proc_macro_derive");
            if let Some(arg) = derive.tt_values().next() {
                if let [TokenTree::Leaf(Leaf::Ident(trait_name)), ..] = &*arg.token_trees {
                    trait_name.as_name()
                } else {
                    log::trace!("malformed `#[proc_macro_derive]`: {}", arg);
                    return;
                }
            } else {
                return;
            }
        };

        self.def_collector.resolve_proc_macro(&macro_name);
    }

    fn collect_macro_rules(&mut self, id: FileItemTreeId<MacroRules>) {
        let krate = self.def_collector.def_map.krate;
        let mac = &self.item_tree[id];
        let attrs = self.item_tree.attrs(self.def_collector.db, krate, ModItem::from(id).into());
        let ast_id = InFile::new(self.file_id, mac.ast_id.upcast());

        let export_attr = attrs.by_key("macro_export");

        let is_export = export_attr.exists();
        let is_local_inner = if is_export {
            export_attr.tt_values().map(|it| &it.token_trees).flatten().any(|it| match it {
                tt::TokenTree::Leaf(tt::Leaf::Ident(ident)) => {
                    ident.text.contains("local_inner_macros")
                }
                _ => false,
            })
        } else {
            false
        };

        // Case 1: builtin macros
        if attrs.by_key("rustc_builtin_macro").exists() {
            let krate = self.def_collector.def_map.krate;
            if let Some(macro_id) = find_builtin_macro(&mac.name, krate, ast_id) {
                self.def_collector.define_macro(
                    self.module_id,
                    mac.name.clone(),
                    macro_id,
                    is_export,
                );
                return;
            }
        }

        // Case 2: normal `macro_rules!` macro
        let macro_id = MacroDefId {
            ast_id: Some(ast_id),
            krate: self.def_collector.def_map.krate,
            kind: MacroDefKind::Declarative,
            local_inner: is_local_inner,
        };
        self.def_collector.define_macro(self.module_id, mac.name.clone(), macro_id, is_export);
    }

    fn collect_macro_call(&mut self, mac: &MacroCall) {
        let mut ast_id = AstIdWithPath::new(self.file_id, mac.ast_id, mac.path.clone());

        // Case 1: try to resolve in legacy scope and expand macro_rules
        let mut error = None;
        match macro_call_as_call_id(
            &ast_id,
            self.def_collector.db,
            self.def_collector.def_map.krate,
            |path| {
                path.as_ident().and_then(|name| {
                    self.def_collector.def_map.with_ancestor_maps(
                        self.def_collector.db,
                        self.module_id,
                        &mut |map, module| map[module].scope.get_legacy_macro(&name),
                    )
                })
            },
            &mut |err| error = Some(err),
        ) {
            Ok(Ok(macro_call_id)) => {
                self.def_collector.unexpanded_macros.push(MacroDirective {
                    module_id: self.module_id,
                    ast_id,
                    legacy: Some(macro_call_id),
                    depth: self.macro_depth + 1,
                });

                return;
            }
            Ok(Err(_)) => {
                // Built-in macro failed eager expansion.
                self.def_collector.def_map.diagnostics.push(DefDiagnostic::macro_error(
                    self.module_id,
                    MacroCallKind::FnLike(ast_id.ast_id),
                    error.map(|e| e.to_string()).unwrap_or_else(|| String::from("macro error")),
                ));
                return;
            }
            Err(UnresolvedMacro) => (),
        }

        // Case 2: resolve in module scope, expand during name resolution.
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

    fn is_cfg_enabled(&self, cfg: &CfgExpr) -> bool {
        self.def_collector.cfg_options.check(cfg) != Some(false)
    }

    fn emit_unconfigured_diagnostic(&mut self, item: ModItem, cfg: &CfgExpr) {
        let ast_id = item.ast_id(self.item_tree);

        let ast_id = InFile::new(self.file_id, ast_id);
        self.def_collector.def_map.diagnostics.push(DefDiagnostic::unconfigured_code(
            self.module_id,
            ast_id,
            cfg.clone(),
            self.def_collector.cfg_options.clone(),
        ));
    }
}

#[cfg(test)]
mod tests {
    use crate::{db::DefDatabase, test_db::TestDB};
    use base_db::{fixture::WithFixture, SourceDatabase};

    use super::*;

    fn do_collect_defs(db: &dyn DefDatabase, def_map: DefMap) -> DefMap {
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
            exports_proc_macros: false,
            from_glob_import: Default::default(),
        };
        collector.seed_with_top_level();
        collector.collect();
        collector.def_map
    }

    fn do_resolve(code: &str) -> DefMap {
        let (db, _file_id) = TestDB::with_single_file(&code);
        let krate = db.test_crate();

        let edition = db.crate_graph()[krate].edition;
        let def_map = DefMap::empty(krate, edition);
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
