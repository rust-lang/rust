//! The core of the module-level name resolution algorithm.
//!
//! `DefCollector::collect` contains the fixed-point iteration loop which
//! resolves imports and expands macros.

use std::{cmp::Ordering, iter, mem};

use base_db::{CrateId, Dependency, Edition, FileId};
use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{
    ast_id_map::FileAstId,
    attrs::{Attr, AttrId},
    builtin_attr_macro::find_builtin_attr,
    builtin_derive_macro::find_builtin_derive,
    builtin_fn_macro::find_builtin_macro,
    hygiene::Hygiene,
    name::{name, AsName, Name},
    proc_macro::ProcMacroExpander,
    ExpandResult, ExpandTo, HirFileId, InFile, MacroCallId, MacroCallKind, MacroCallLoc,
    MacroDefId, MacroDefKind,
};
use itertools::{izip, Itertools};
use la_arena::Idx;
use limit::Limit;
use rustc_hash::{FxHashMap, FxHashSet};
use stdx::always;
use syntax::{ast, SmolStr};
use triomphe::Arc;

use crate::{
    attr::Attrs,
    attr_macro_as_call_id,
    db::DefDatabase,
    derive_macro_as_call_id,
    item_scope::{ImportType, PerNsGlobImports},
    item_tree::{
        self, ExternCrate, Fields, FileItemTreeId, ImportKind, ItemTree, ItemTreeId, ItemTreeNode,
        MacroCall, MacroDef, MacroRules, Mod, ModItem, ModKind, TreeId,
    },
    macro_call_as_call_id, macro_id_to_def_id,
    nameres::{
        diagnostics::DefDiagnostic,
        mod_resolution::ModDir,
        path_resolution::ReachedFixedPoint,
        proc_macro::{parse_macro_name_and_helper_attrs, ProcMacroDef, ProcMacroKind},
        sub_namespace_match, BuiltinShadowMode, DefMap, MacroSubNs, ModuleData, ModuleOrigin,
        ResolveMode,
    },
    path::{ImportAlias, ModPath, PathKind},
    per_ns::PerNs,
    tt,
    visibility::{RawVisibility, Visibility},
    AdtId, AstId, AstIdWithPath, ConstLoc, CrateRootModuleId, EnumLoc, EnumVariantId,
    ExternBlockLoc, FunctionId, FunctionLoc, ImplLoc, Intern, ItemContainerId, LocalModuleId,
    Macro2Id, Macro2Loc, MacroExpander, MacroId, MacroRulesId, MacroRulesLoc, ModuleDefId,
    ModuleId, ProcMacroId, ProcMacroLoc, StaticLoc, StructLoc, TraitAliasLoc, TraitLoc,
    TypeAliasLoc, UnionLoc, UnresolvedMacro,
};

static GLOB_RECURSION_LIMIT: Limit = Limit::new(100);
static EXPANSION_DEPTH_LIMIT: Limit = Limit::new(128);
static FIXED_POINT_LIMIT: Limit = Limit::new(8192);

pub(super) fn collect_defs(db: &dyn DefDatabase, def_map: DefMap, tree_id: TreeId) -> DefMap {
    let crate_graph = db.crate_graph();

    let mut deps = FxHashMap::default();
    // populate external prelude and dependency list
    let krate = &crate_graph[def_map.krate];
    for dep in &krate.dependencies {
        tracing::debug!("crate dep {:?} -> {:?}", dep.name, dep.crate_id);

        deps.insert(dep.as_name(), dep.clone());
    }

    let cfg_options = &krate.cfg_options;

    let is_proc_macro = krate.is_proc_macro;
    let proc_macros = if is_proc_macro {
        match db.proc_macros().get(&def_map.krate) {
            Some(Ok(proc_macros)) => {
                Ok(proc_macros
                    .iter()
                    .enumerate()
                    .map(|(idx, it)| {
                        // FIXME: a hacky way to create a Name from string.
                        let name =
                            tt::Ident { text: it.name.clone(), span: tt::TokenId::unspecified() };
                        (name.as_name(), ProcMacroExpander::new(base_db::ProcMacroId(idx as u32)))
                    })
                    .collect())
            }
            Some(Err(e)) => Err(e.clone().into_boxed_str()),
            None => Err("No proc-macros present for crate".to_owned().into_boxed_str()),
        }
    } else {
        Ok(vec![])
    };

    let mut collector = DefCollector {
        db,
        def_map,
        deps,
        glob_imports: FxHashMap::default(),
        unresolved_imports: Vec::new(),
        indeterminate_imports: Vec::new(),
        unresolved_macros: Vec::new(),
        mod_dirs: FxHashMap::default(),
        cfg_options,
        proc_macros,
        from_glob_import: Default::default(),
        skip_attrs: Default::default(),
        is_proc_macro,
        hygienes: FxHashMap::default(),
    };
    if tree_id.is_block() {
        collector.seed_with_inner(tree_id);
    } else {
        collector.seed_with_top_level();
    }
    collector.collect();
    let mut def_map = collector.finish();
    def_map.shrink_to_fit();
    def_map
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum PartialResolvedImport {
    /// None of any namespaces is resolved
    Unresolved,
    /// One of namespaces is resolved
    Indeterminate(PerNs),
    /// All namespaces are resolved, OR it comes from other crate
    Resolved(PerNs),
}

impl PartialResolvedImport {
    fn namespaces(self) -> PerNs {
        match self {
            PartialResolvedImport::Unresolved => PerNs::none(),
            PartialResolvedImport::Indeterminate(ns) | PartialResolvedImport::Resolved(ns) => ns,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ImportSource {
    Import { id: ItemTreeId<item_tree::Import>, use_tree: Idx<ast::UseTree> },
    ExternCrate(ItemTreeId<item_tree::ExternCrate>),
}

#[derive(Debug, Eq, PartialEq)]
struct Import {
    path: ModPath,
    alias: Option<ImportAlias>,
    visibility: RawVisibility,
    kind: ImportKind,
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
    ) -> Vec<Self> {
        let it = &tree[id.value];
        let attrs = &tree.attrs(db, krate, ModItem::from(id.value).into());
        let visibility = &tree[it.visibility];
        let is_prelude = attrs.by_key("prelude_import").exists();

        let mut res = Vec::new();
        it.use_tree.expand(|idx, path, kind, alias| {
            res.push(Self {
                path,
                alias,
                visibility: visibility.clone(),
                kind,
                is_prelude,
                is_extern_crate: false,
                is_macro_use: false,
                source: ImportSource::Import { id, use_tree: idx },
            });
        });
        res
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
            kind: ImportKind::Plain,
            is_prelude: false,
            is_extern_crate: true,
            is_macro_use: attrs.by_key("macro_use").exists(),
            source: ImportSource::ExternCrate(id),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct ImportDirective {
    /// The module this import directive is in.
    module_id: LocalModuleId,
    import: Import,
    status: PartialResolvedImport,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MacroDirective {
    module_id: LocalModuleId,
    depth: usize,
    kind: MacroDirectiveKind,
    container: ItemContainerId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum MacroDirectiveKind {
    FnLike { ast_id: AstIdWithPath<ast::MacroCall>, expand_to: ExpandTo },
    Derive { ast_id: AstIdWithPath<ast::Adt>, derive_attr: AttrId, derive_pos: usize },
    Attr { ast_id: AstIdWithPath<ast::Item>, attr: Attr, mod_item: ModItem, tree: TreeId },
}

/// Walks the tree of module recursively
struct DefCollector<'a> {
    db: &'a dyn DefDatabase,
    def_map: DefMap,
    deps: FxHashMap<Name, Dependency>,
    glob_imports: FxHashMap<LocalModuleId, Vec<(LocalModuleId, Visibility)>>,
    unresolved_imports: Vec<ImportDirective>,
    indeterminate_imports: Vec<ImportDirective>,
    unresolved_macros: Vec<MacroDirective>,
    mod_dirs: FxHashMap<LocalModuleId, ModDir>,
    cfg_options: &'a CfgOptions,
    /// List of procedural macros defined by this crate. This is read from the dynamic library
    /// built by the build system, and is the list of proc. macros we can actually expand. It is
    /// empty when proc. macro support is disabled (in which case we still do name resolution for
    /// them).
    proc_macros: Result<Vec<(Name, ProcMacroExpander)>, Box<str>>,
    is_proc_macro: bool,
    from_glob_import: PerNsGlobImports,
    /// If we fail to resolve an attribute on a `ModItem`, we fall back to ignoring the attribute.
    /// This map is used to skip all attributes up to and including the one that failed to resolve,
    /// in order to not expand them twice.
    ///
    /// This also stores the attributes to skip when we resolve derive helpers and non-macro
    /// non-builtin attributes in general.
    skip_attrs: FxHashMap<InFile<ModItem>, AttrId>,
    /// `Hygiene` cache, because `Hygiene` construction is expensive.
    ///
    /// Almost all paths should have been lowered to `ModPath` during `ItemTree` construction.
    /// However, `DefCollector` still needs to lower paths in attributes, in particular those in
    /// derive meta item list.
    hygienes: FxHashMap<HirFileId, Hygiene>,
}

impl DefCollector<'_> {
    fn seed_with_top_level(&mut self) {
        let _p = profile::span("seed_with_top_level");

        let file_id = self.db.crate_graph()[self.def_map.krate].root_file_id;
        let item_tree = self.db.file_item_tree(file_id.into());
        let attrs = item_tree.top_level_attrs(self.db, self.def_map.krate);
        let crate_data = Arc::get_mut(&mut self.def_map.data).unwrap();

        if let Err(e) = &self.proc_macros {
            crate_data.proc_macro_loading_error = Some(e.clone());
        }

        for (name, dep) in &self.deps {
            if dep.is_prelude() {
                crate_data
                    .extern_prelude
                    .insert(name.clone(), CrateRootModuleId { krate: dep.crate_id });
            }
        }

        // Process other crate-level attributes.
        for attr in &*attrs {
            if let Some(cfg) = attr.cfg() {
                if self.cfg_options.check(&cfg) == Some(false) {
                    return;
                }
            }
            let attr_name = match attr.path.as_ident() {
                Some(name) => name,
                None => continue,
            };

            if *attr_name == hir_expand::name![recursion_limit] {
                if let Some(limit) = attr.string_value() {
                    if let Ok(limit) = limit.parse() {
                        crate_data.recursion_limit = Some(limit);
                    }
                }
                continue;
            }

            if *attr_name == hir_expand::name![crate_type] {
                if let Some("proc-macro") = attr.string_value().map(SmolStr::as_str) {
                    self.is_proc_macro = true;
                }
                continue;
            }

            if *attr_name == hir_expand::name![no_core] {
                crate_data.no_core = true;
                continue;
            }

            if *attr_name == hir_expand::name![no_std] {
                crate_data.no_std = true;
                continue;
            }

            if attr_name.as_text().as_deref() == Some("rustc_coherence_is_core") {
                crate_data.rustc_coherence_is_core = true;
                continue;
            }

            if *attr_name == hir_expand::name![feature] {
                let hygiene = &Hygiene::new_unhygienic();
                let features = attr
                    .parse_path_comma_token_tree(self.db.upcast(), hygiene)
                    .into_iter()
                    .flatten()
                    .filter_map(|feat| match feat.segments() {
                        [name] => Some(name.to_smol_str()),
                        _ => None,
                    });
                crate_data.unstable_features.extend(features);
            }

            let attr_is_register_like = *attr_name == hir_expand::name![register_attr]
                || *attr_name == hir_expand::name![register_tool];
            if !attr_is_register_like {
                continue;
            }

            let registered_name = match attr.single_ident_value() {
                Some(ident) => ident.as_name(),
                _ => continue,
            };

            if *attr_name == hir_expand::name![register_attr] {
                crate_data.registered_attrs.push(registered_name.to_smol_str());
                cov_mark::hit!(register_attr);
            } else {
                crate_data.registered_tools.push(registered_name.to_smol_str());
                cov_mark::hit!(register_tool);
            }
        }

        crate_data.shrink_to_fit();
        self.inject_prelude();

        ModCollector {
            def_collector: self,
            macro_depth: 0,
            module_id: DefMap::ROOT,
            tree_id: TreeId::new(file_id.into(), None),
            item_tree: &item_tree,
            mod_dir: ModDir::root(),
        }
        .collect_in_top_module(item_tree.top_level_items());
    }

    fn seed_with_inner(&mut self, tree_id: TreeId) {
        let item_tree = tree_id.item_tree(self.db);
        let is_cfg_enabled = item_tree
            .top_level_attrs(self.db, self.def_map.krate)
            .cfg()
            .map_or(true, |cfg| self.cfg_options.check(&cfg) != Some(false));
        if is_cfg_enabled {
            ModCollector {
                def_collector: self,
                macro_depth: 0,
                module_id: DefMap::ROOT,
                tree_id,
                item_tree: &item_tree,
                mod_dir: ModDir::root(),
            }
            .collect_in_top_module(item_tree.top_level_items());
        }
    }

    fn resolution_loop(&mut self) {
        let _p = profile::span("DefCollector::resolution_loop");

        // main name resolution fixed-point loop.
        let mut i = 0;
        'resolve_attr: loop {
            'resolve_macros: loop {
                self.db.unwind_if_cancelled();

                {
                    let _p = profile::span("resolve_imports loop");

                    'resolve_imports: loop {
                        if self.resolve_imports() == ReachedFixedPoint::Yes {
                            break 'resolve_imports;
                        }
                    }
                }
                if self.resolve_macros() == ReachedFixedPoint::Yes {
                    break 'resolve_macros;
                }

                i += 1;
                if FIXED_POINT_LIMIT.check(i).is_err() {
                    tracing::error!("name resolution is stuck");
                    break 'resolve_attr;
                }
            }

            if self.reseed_with_unresolved_attribute() == ReachedFixedPoint::Yes {
                break 'resolve_attr;
            }
        }
    }

    fn collect(&mut self) {
        let _p = profile::span("DefCollector::collect");

        self.resolution_loop();

        // Resolve all indeterminate resolved imports again
        // As some of the macros will expand newly import shadowing partial resolved imports
        // FIXME: We maybe could skip this, if we handle the indeterminate imports in `resolve_imports`
        // correctly
        let partial_resolved = self.indeterminate_imports.drain(..).map(|directive| {
            ImportDirective { status: PartialResolvedImport::Unresolved, ..directive }
        });
        self.unresolved_imports.extend(partial_resolved);
        self.resolve_imports();

        let unresolved_imports = mem::take(&mut self.unresolved_imports);
        // show unresolved imports in completion, etc
        for directive in &unresolved_imports {
            self.record_resolved_import(directive);
        }
        self.unresolved_imports = unresolved_imports;

        if self.is_proc_macro {
            // A crate exporting procedural macros is not allowed to export anything else.
            //
            // Additionally, while the proc macro entry points must be `pub`, they are not publicly
            // exported in type/value namespace. This function reduces the visibility of all items
            // in the crate root that aren't proc macros.
            let root = DefMap::ROOT;
            let module_id = self.def_map.module_id(root);
            let root = &mut self.def_map.modules[root];
            root.scope.censor_non_proc_macros(module_id);
        }
    }

    /// When the fixed-point loop reaches a stable state, we might still have
    /// some unresolved attributes left over. This takes one of them, and feeds
    /// the item it's applied to back into name resolution.
    ///
    /// This effectively ignores the fact that the macro is there and just treats the items as
    /// normal code.
    ///
    /// This improves UX for unresolved attributes, and replicates the
    /// behavior before we supported proc. attribute macros.
    fn reseed_with_unresolved_attribute(&mut self) -> ReachedFixedPoint {
        cov_mark::hit!(unresolved_attribute_fallback);

        let unresolved_attr =
            self.unresolved_macros.iter().enumerate().find_map(|(idx, directive)| match &directive
                .kind
            {
                MacroDirectiveKind::Attr { ast_id, mod_item, attr, tree } => {
                    self.def_map.diagnostics.push(DefDiagnostic::unresolved_macro_call(
                        directive.module_id,
                        MacroCallKind::Attr {
                            ast_id: ast_id.ast_id,
                            attr_args: Arc::new((tt::Subtree::empty(), Default::default())),
                            invoc_attr_index: attr.id,
                        },
                        attr.path().clone(),
                    ));

                    self.skip_attrs.insert(ast_id.ast_id.with_value(*mod_item), attr.id);

                    Some((idx, directive, *mod_item, *tree))
                }
                _ => None,
            });

        match unresolved_attr {
            Some((pos, &MacroDirective { module_id, depth, container, .. }, mod_item, tree_id)) => {
                let item_tree = &tree_id.item_tree(self.db);
                let mod_dir = self.mod_dirs[&module_id].clone();
                ModCollector {
                    def_collector: self,
                    macro_depth: depth,
                    module_id,
                    tree_id,
                    item_tree,
                    mod_dir,
                }
                .collect(&[mod_item], container);

                self.unresolved_macros.swap_remove(pos);
                // Continue name resolution with the new data.
                ReachedFixedPoint::No
            }
            None => ReachedFixedPoint::Yes,
        }
    }

    fn inject_prelude(&mut self) {
        // See compiler/rustc_builtin_macros/src/standard_library_imports.rs

        if self.def_map.data.no_core {
            // libcore does not get a prelude.
            return;
        }

        let krate = if self.def_map.data.no_std {
            name![core]
        } else {
            let std = name![std];
            if self.def_map.extern_prelude().any(|(name, _)| *name == std) {
                std
            } else {
                // If `std` does not exist for some reason, fall back to core. This mostly helps
                // keep r-a's own tests minimal.
                name![core]
            }
        };

        let edition = match self.def_map.data.edition {
            Edition::Edition2015 => name![rust_2015],
            Edition::Edition2018 => name![rust_2018],
            Edition::Edition2021 => name![rust_2021],
        };

        let path_kind = match self.def_map.data.edition {
            Edition::Edition2015 => PathKind::Plain,
            _ => PathKind::Abs,
        };
        let path = ModPath::from_segments(path_kind, [krate, name![prelude], edition]);

        let (per_ns, _) =
            self.def_map.resolve_path(self.db, DefMap::ROOT, &path, BuiltinShadowMode::Other, None);

        match per_ns.types {
            Some((ModuleDefId::ModuleId(m), _)) => {
                self.def_map.prelude = Some(m);
            }
            types => {
                tracing::debug!(
                    "could not resolve prelude path `{}` to module (resolved to {:?})",
                    path.display(self.db.upcast()),
                    types
                );
            }
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
    fn export_proc_macro(
        &mut self,
        def: ProcMacroDef,
        id: ItemTreeId<item_tree::Function>,
        fn_id: FunctionId,
    ) {
        if self.def_map.block.is_some() {
            return;
        }
        let kind = def.kind.to_basedb_kind();
        let (expander, kind) =
            match self.proc_macros.as_ref().map(|it| it.iter().find(|(n, _)| n == &def.name)) {
                Ok(Some(&(_, expander))) => (expander, kind),
                _ => (ProcMacroExpander::dummy(), kind),
            };

        let proc_macro_id =
            ProcMacroLoc { container: self.def_map.crate_root(), id, expander, kind }
                .intern(self.db);
        self.define_proc_macro(def.name.clone(), proc_macro_id);
        let crate_data = Arc::get_mut(&mut self.def_map.data).unwrap();
        if let ProcMacroKind::CustomDerive { helpers } = def.kind {
            crate_data
                .exported_derives
                .insert(macro_id_to_def_id(self.db, proc_macro_id.into()), helpers);
        }
        crate_data.fn_proc_macro_mapping.insert(fn_id, proc_macro_id);
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
    fn define_macro_rules(
        &mut self,
        module_id: LocalModuleId,
        name: Name,
        macro_: MacroRulesId,
        export: bool,
    ) {
        // Textual scoping
        self.define_legacy_macro(module_id, name.clone(), macro_.into());

        // Module scoping
        // In Rust, `#[macro_export]` macros are unconditionally visible at the
        // crate root, even if the parent modules is **not** visible.
        if export {
            let module_id = DefMap::ROOT;
            self.def_map.modules[module_id].scope.declare(macro_.into());
            self.update(
                module_id,
                &[(Some(name), PerNs::macros(macro_.into(), Visibility::Public))],
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
    fn define_legacy_macro(&mut self, module_id: LocalModuleId, name: Name, mac: MacroId) {
        // Always shadowing
        self.def_map.modules[module_id].scope.define_legacy_macro(name, mac);
    }

    /// Define a macro 2.0 macro
    ///
    /// The scoped of macro 2.0 macro is equal to normal function
    fn define_macro_def(
        &mut self,
        module_id: LocalModuleId,
        name: Name,
        macro_: Macro2Id,
        vis: &RawVisibility,
    ) {
        let vis = self
            .def_map
            .resolve_visibility(self.db, module_id, vis, false)
            .unwrap_or(Visibility::Public);
        self.def_map.modules[module_id].scope.declare(macro_.into());
        self.update(
            module_id,
            &[(Some(name), PerNs::macros(macro_.into(), Visibility::Public))],
            vis,
            ImportType::Named,
        );
    }

    /// Define a proc macro
    ///
    /// A proc macro is similar to normal macro scope, but it would not visible in legacy textual scoped.
    /// And unconditionally exported.
    fn define_proc_macro(&mut self, name: Name, macro_: ProcMacroId) {
        let module_id = DefMap::ROOT;
        self.def_map.modules[module_id].scope.declare(macro_.into());
        self.update(
            module_id,
            &[(Some(name), PerNs::macros(macro_.into(), Visibility::Public))],
            Visibility::Public,
            ImportType::Named,
        );
    }

    /// Import exported macros from another crate. `names`, if `Some(_)`, specifies the name of
    /// macros to be imported. Otherwise this method imports all exported macros.
    ///
    /// Exported macros are just all macros in the root module scope.
    /// Note that it contains not only all `#[macro_export]` macros, but also all aliases
    /// created by `use` in the root module, ignoring the visibility of `use`.
    fn import_macros_from_extern_crate(&mut self, krate: CrateId, names: Option<Vec<Name>>) {
        let def_map = self.db.crate_def_map(krate);
        // `#[macro_use]` brings macros into macro_use prelude. Yes, even non-`macro_rules!`
        // macros.
        let root_scope = &def_map[DefMap::ROOT].scope;
        if let Some(names) = names {
            for name in names {
                // FIXME: Report diagnostic on 404.
                if let Some(def) = root_scope.get(&name).take_macros() {
                    self.def_map.macro_use_prelude.insert(name, def);
                }
            }
        } else {
            for (name, def) in root_scope.macros() {
                self.def_map.macro_use_prelude.insert(name.clone(), def);
            }
        }
    }

    /// Tries to resolve every currently unresolved import.
    fn resolve_imports(&mut self) -> ReachedFixedPoint {
        let mut res = ReachedFixedPoint::Yes;
        let imports = mem::take(&mut self.unresolved_imports);

        self.unresolved_imports = imports
            .into_iter()
            .filter_map(|mut directive| {
                directive.status = self.resolve_import(directive.module_id, &directive.import);
                match directive.status {
                    PartialResolvedImport::Indeterminate(_) => {
                        self.record_resolved_import(&directive);
                        self.indeterminate_imports.push(directive);
                        res = ReachedFixedPoint::No;
                        None
                    }
                    PartialResolvedImport::Resolved(_) => {
                        self.record_resolved_import(&directive);
                        res = ReachedFixedPoint::No;
                        None
                    }
                    PartialResolvedImport::Unresolved => Some(directive),
                }
            })
            .collect();
        res
    }

    fn resolve_import(&self, module_id: LocalModuleId, import: &Import) -> PartialResolvedImport {
        let _p = profile::span("resolve_import")
            .detail(|| format!("{}", import.path.display(self.db.upcast())));
        tracing::debug!("resolving import: {:?} ({:?})", import, self.def_map.data.edition);
        if import.is_extern_crate {
            let name = import
                .path
                .as_ident()
                .expect("extern crate should have been desugared to one-element path");

            let res = self.resolve_extern_crate(name);

            match res {
                Some(res) => {
                    PartialResolvedImport::Resolved(PerNs::types(res.into(), Visibility::Public))
                }
                None => PartialResolvedImport::Unresolved,
            }
        } else {
            let res = self.def_map.resolve_path_fp_with_macro(
                self.db,
                ResolveMode::Import,
                module_id,
                &import.path,
                BuiltinShadowMode::Module,
                None, // An import may resolve to any kind of macro.
            );

            let def = res.resolved_def;
            if res.reached_fixedpoint == ReachedFixedPoint::No || def.is_none() {
                return PartialResolvedImport::Unresolved;
            }

            if let Some(krate) = res.krate {
                if krate != self.def_map.krate {
                    return PartialResolvedImport::Resolved(
                        def.filter_visibility(|v| matches!(v, Visibility::Public)),
                    );
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

    fn resolve_extern_crate(&self, name: &Name) -> Option<CrateRootModuleId> {
        if *name == name!(self) {
            cov_mark::hit!(extern_crate_self_as);
            Some(self.def_map.crate_root())
        } else {
            self.deps.get(name).map(|dep| CrateRootModuleId { krate: dep.crate_id })
        }
    }

    fn record_resolved_import(&mut self, directive: &ImportDirective) {
        let _p = profile::span("record_resolved_import");

        let module_id = directive.module_id;
        let import = &directive.import;
        let mut def = directive.status.namespaces();
        let vis = self
            .def_map
            .resolve_visibility(self.db, module_id, &directive.import.visibility, false)
            .unwrap_or(Visibility::Public);

        match import.kind {
            ImportKind::Plain | ImportKind::TypeOnly => {
                let name = match &import.alias {
                    Some(ImportAlias::Alias(name)) => Some(name),
                    Some(ImportAlias::Underscore) => None,
                    None => match import.path.segments().last() {
                        Some(last_segment) => Some(last_segment),
                        None => {
                            cov_mark::hit!(bogus_paths);
                            return;
                        }
                    },
                };

                if import.kind == ImportKind::TypeOnly {
                    def.values = None;
                    def.macros = None;
                }

                tracing::debug!("resolved import {:?} ({:?}) to {:?}", name, import, def);

                // extern crates in the crate root are special-cased to insert entries into the extern prelude: rust-lang/rust#54658
                if import.is_extern_crate
                    && self.def_map.block.is_none()
                    && module_id == DefMap::ROOT
                {
                    if let (Some(ModuleDefId::ModuleId(def)), Some(name)) = (def.take_types(), name)
                    {
                        if let Ok(def) = def.try_into() {
                            Arc::get_mut(&mut self.def_map.data)
                                .unwrap()
                                .extern_prelude
                                .insert(name.clone(), def);
                        }
                    }
                }

                self.update(module_id, &[(name.cloned(), def)], vis, ImportType::Named);
            }
            ImportKind::Glob => {
                tracing::debug!("glob import: {:?}", import);
                match def.take_types() {
                    Some(ModuleDefId::ModuleId(m)) => {
                        if import.is_prelude {
                            // Note: This dodgily overrides the injected prelude. The rustc
                            // implementation seems to work the same though.
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
                                            v.is_visible_from_def_map(
                                                self.db,
                                                &self.def_map,
                                                module_id,
                                            )
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
                        tracing::debug!("glob import {:?} from non-module/enum {:?}", import, d);
                    }
                    None => {
                        tracing::debug!("glob import {:?} didn't resolve as type", import);
                    }
                }
            }
        }
    }

    fn update(
        &mut self,
        // The module for which `resolutions` have been resolve
        module_id: LocalModuleId,
        resolutions: &[(Option<Name>, PerNs)],
        // Visibility this import will have
        vis: Visibility,
        import_type: ImportType,
    ) {
        self.db.unwind_if_cancelled();
        self.update_recursive(module_id, resolutions, vis, import_type, 0)
    }

    fn update_recursive(
        &mut self,
        // The module for which `resolutions` have been resolve
        module_id: LocalModuleId,
        resolutions: &[(Option<Name>, PerNs)],
        // All resolutions are imported with this visibility; the visibilities in
        // the `PerNs` values are ignored and overwritten
        vis: Visibility,
        import_type: ImportType,
        depth: usize,
    ) {
        if GLOB_RECURSION_LIMIT.check(depth).is_err() {
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
                            tracing::debug!("non-trait `_` import of {:?}", other);
                            continue;
                        }
                        None => continue,
                    };
                    let old_vis = self.def_map.modules[module_id].scope.unnamed_trait_vis(tr);
                    let should_update = match old_vis {
                        None => true,
                        Some(old_vis) => {
                            let max_vis = old_vis.max(vis, &self.def_map).unwrap_or_else(|| {
                                panic!("`Tr as _` imports with unrelated visibilities {old_vis:?} and {vis:?} (trait {tr:?})");
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
            .flatten()
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
        let mut macros = mem::take(&mut self.unresolved_macros);
        let mut resolved = Vec::new();
        let mut push_resolved = |directive: &MacroDirective, call_id| {
            resolved.push((directive.module_id, directive.depth, directive.container, call_id));
        };
        let mut res = ReachedFixedPoint::Yes;
        // Retain unresolved macros after this round of resolution.
        macros.retain(|directive| {
            let subns = match &directive.kind {
                MacroDirectiveKind::FnLike { .. } => MacroSubNs::Bang,
                MacroDirectiveKind::Attr { .. } | MacroDirectiveKind::Derive { .. } => {
                    MacroSubNs::Attr
                }
            };
            let resolver = |path| {
                let resolved_res = self.def_map.resolve_path_fp_with_macro(
                    self.db,
                    ResolveMode::Other,
                    directive.module_id,
                    &path,
                    BuiltinShadowMode::Module,
                    Some(subns),
                );
                resolved_res
                    .resolved_def
                    .take_macros()
                    .map(|it| (it, macro_id_to_def_id(self.db, it)))
            };
            let resolver_def_id = |path| resolver(path).map(|(_, it)| it);

            match &directive.kind {
                MacroDirectiveKind::FnLike { ast_id, expand_to } => {
                    let call_id = macro_call_as_call_id(
                        self.db.upcast(),
                        ast_id,
                        *expand_to,
                        self.def_map.krate,
                        resolver_def_id,
                    );
                    if let Ok(Some(call_id)) = call_id {
                        push_resolved(directive, call_id);

                        res = ReachedFixedPoint::No;
                        return false;
                    }
                }
                MacroDirectiveKind::Derive { ast_id, derive_attr, derive_pos } => {
                    let id = derive_macro_as_call_id(
                        self.db,
                        ast_id,
                        *derive_attr,
                        *derive_pos as u32,
                        self.def_map.krate,
                        resolver,
                    );

                    if let Ok((macro_id, def_id, call_id)) = id {
                        self.def_map.modules[directive.module_id].scope.set_derive_macro_invoc(
                            ast_id.ast_id,
                            call_id,
                            *derive_attr,
                            *derive_pos,
                        );
                        // Record its helper attributes.
                        if def_id.krate != self.def_map.krate {
                            let def_map = self.db.crate_def_map(def_id.krate);
                            if let Some(helpers) = def_map.data.exported_derives.get(&def_id) {
                                self.def_map
                                    .derive_helpers_in_scope
                                    .entry(ast_id.ast_id.map(|it| it.upcast()))
                                    .or_default()
                                    .extend(izip!(
                                        helpers.iter().cloned(),
                                        iter::repeat(macro_id),
                                        iter::repeat(call_id),
                                    ));
                            }
                        }

                        push_resolved(directive, call_id);
                        res = ReachedFixedPoint::No;
                        return false;
                    }
                }
                MacroDirectiveKind::Attr { ast_id: file_ast_id, mod_item, attr, tree } => {
                    let &AstIdWithPath { ast_id, ref path } = file_ast_id;
                    let file_id = ast_id.file_id;

                    let mut recollect_without = |collector: &mut Self| {
                        // Remove the original directive since we resolved it.
                        let mod_dir = collector.mod_dirs[&directive.module_id].clone();
                        collector.skip_attrs.insert(InFile::new(file_id, *mod_item), attr.id);

                        let item_tree = tree.item_tree(self.db);
                        ModCollector {
                            def_collector: collector,
                            macro_depth: directive.depth,
                            module_id: directive.module_id,
                            tree_id: *tree,
                            item_tree: &item_tree,
                            mod_dir,
                        }
                        .collect(&[*mod_item], directive.container);
                        res = ReachedFixedPoint::No;
                        false
                    };

                    if let Some(ident) = path.as_ident() {
                        if let Some(helpers) = self.def_map.derive_helpers_in_scope.get(&ast_id) {
                            if helpers.iter().any(|(it, ..)| it == ident) {
                                cov_mark::hit!(resolved_derive_helper);
                                // Resolved to derive helper. Collect the item's attributes again,
                                // starting after the derive helper.
                                return recollect_without(self);
                            }
                        }
                    }

                    let def = match resolver_def_id(path.clone()) {
                        Some(def) if def.is_attribute() => def,
                        _ => return true,
                    };
                    if matches!(
                        def,
                        MacroDefId { kind:MacroDefKind::BuiltInAttr(expander, _),.. }
                        if expander.is_derive()
                    ) {
                        // Resolved to `#[derive]`

                        let item_tree = tree.item_tree(self.db);
                        let ast_adt_id: FileAstId<ast::Adt> = match *mod_item {
                            ModItem::Struct(strukt) => item_tree[strukt].ast_id().upcast(),
                            ModItem::Union(union) => item_tree[union].ast_id().upcast(),
                            ModItem::Enum(enum_) => item_tree[enum_].ast_id().upcast(),
                            _ => {
                                let diag = DefDiagnostic::invalid_derive_target(
                                    directive.module_id,
                                    ast_id,
                                    attr.id,
                                );
                                self.def_map.diagnostics.push(diag);
                                return recollect_without(self);
                            }
                        };
                        let ast_id = ast_id.with_value(ast_adt_id);

                        let extend_unhygenic;
                        let hygiene = if file_id.is_macro() {
                            self.hygienes
                                .entry(file_id)
                                .or_insert_with(|| Hygiene::new(self.db.upcast(), file_id))
                        } else {
                            // Avoid heap allocation (`Hygiene` embraces `Arc`) and hash map entry
                            // when we're in an oridinary (non-macro) file.
                            extend_unhygenic = Hygiene::new_unhygienic();
                            &extend_unhygenic
                        };

                        match attr.parse_path_comma_token_tree(self.db.upcast(), hygiene) {
                            Some(derive_macros) => {
                                let mut len = 0;
                                for (idx, path) in derive_macros.enumerate() {
                                    let ast_id = AstIdWithPath::new(file_id, ast_id.value, path);
                                    self.unresolved_macros.push(MacroDirective {
                                        module_id: directive.module_id,
                                        depth: directive.depth + 1,
                                        kind: MacroDirectiveKind::Derive {
                                            ast_id,
                                            derive_attr: attr.id,
                                            derive_pos: idx,
                                        },
                                        container: directive.container,
                                    });
                                    len = idx;
                                }

                                // We treat the #[derive] macro as an attribute call, but we do not resolve it for nameres collection.
                                // This is just a trick to be able to resolve the input to derives as proper paths.
                                // Check the comment in [`builtin_attr_macro`].
                                let call_id = attr_macro_as_call_id(
                                    self.db,
                                    file_ast_id,
                                    attr,
                                    self.def_map.krate,
                                    def,
                                );
                                self.def_map.modules[directive.module_id]
                                    .scope
                                    .init_derive_attribute(ast_id, attr.id, call_id, len + 1);
                            }
                            None => {
                                let diag = DefDiagnostic::malformed_derive(
                                    directive.module_id,
                                    ast_id,
                                    attr.id,
                                );
                                self.def_map.diagnostics.push(diag);
                            }
                        }

                        return recollect_without(self);
                    }

                    // Not resolved to a derive helper or the derive attribute, so try to treat as a normal attribute.
                    let call_id =
                        attr_macro_as_call_id(self.db, file_ast_id, attr, self.def_map.krate, def);
                    let loc: MacroCallLoc = self.db.lookup_intern_macro_call(call_id);

                    // If proc attribute macro expansion is disabled, skip expanding it here
                    if !self.db.expand_proc_attr_macros() {
                        self.def_map.diagnostics.push(DefDiagnostic::unresolved_proc_macro(
                            directive.module_id,
                            loc.kind,
                            loc.def.krate,
                        ));
                        return recollect_without(self);
                    }

                    // Skip #[test]/#[bench] expansion, which would merely result in more memory usage
                    // due to duplicating functions into macro expansions
                    if matches!(
                        loc.def.kind,
                        MacroDefKind::BuiltInAttr(expander, _)
                        if expander.is_test() || expander.is_bench()
                    ) {
                        return recollect_without(self);
                    }

                    if let MacroDefKind::ProcMacro(exp, ..) = loc.def.kind {
                        if exp.is_dummy() {
                            // If there's no expander for the proc macro (e.g.
                            // because proc macros are disabled, or building the
                            // proc macro crate failed), report this and skip
                            // expansion like we would if it was disabled
                            self.def_map.diagnostics.push(DefDiagnostic::unresolved_proc_macro(
                                directive.module_id,
                                loc.kind,
                                loc.def.krate,
                            ));

                            return recollect_without(self);
                        }
                    }

                    self.def_map.modules[directive.module_id]
                        .scope
                        .add_attr_macro_invoc(ast_id, call_id);

                    push_resolved(directive, call_id);
                    res = ReachedFixedPoint::No;
                    return false;
                }
            }

            true
        });
        // Attribute resolution can add unresolved macro invocations, so concatenate the lists.
        macros.extend(mem::take(&mut self.unresolved_macros));
        self.unresolved_macros = macros;

        for (module_id, depth, container, macro_call_id) in resolved {
            self.collect_macro_expansion(module_id, macro_call_id, depth, container);
        }

        res
    }

    fn collect_macro_expansion(
        &mut self,
        module_id: LocalModuleId,
        macro_call_id: MacroCallId,
        depth: usize,
        container: ItemContainerId,
    ) {
        if EXPANSION_DEPTH_LIMIT.check(depth).is_err() {
            cov_mark::hit!(macro_expansion_overflow);
            tracing::warn!("macro expansion is too deep");
            return;
        }
        let file_id = macro_call_id.as_file();

        // First, fetch the raw expansion result for purposes of error reporting. This goes through
        // `parse_macro_expansion_error` to avoid depending on the full expansion result (to improve
        // incrementality).
        let ExpandResult { value, err } = self.db.parse_macro_expansion_error(macro_call_id);
        if let Some(err) = err {
            let loc: MacroCallLoc = self.db.lookup_intern_macro_call(macro_call_id);
            let diag = match err {
                // why is this reported here?
                hir_expand::ExpandError::UnresolvedProcMacro(krate) => {
                    always!(krate == loc.def.krate);
                    DefDiagnostic::unresolved_proc_macro(module_id, loc.kind.clone(), loc.def.krate)
                }
                _ => DefDiagnostic::macro_error(module_id, loc.kind.clone(), err.to_string()),
            };

            self.def_map.diagnostics.push(diag);
        }
        if let errors @ [_, ..] = &*value {
            let loc: MacroCallLoc = self.db.lookup_intern_macro_call(macro_call_id);
            let diag = DefDiagnostic::macro_expansion_parse_error(module_id, loc.kind, &errors);
            self.def_map.diagnostics.push(diag);
        }

        // Then, fetch and process the item tree. This will reuse the expansion result from above.
        let item_tree = self.db.file_item_tree(file_id);

        let mod_dir = self.mod_dirs[&module_id].clone();
        ModCollector {
            def_collector: &mut *self,
            macro_depth: depth,
            tree_id: TreeId::new(file_id, None),
            module_id,
            item_tree: &item_tree,
            mod_dir,
        }
        .collect(item_tree.top_level_items(), container);
    }

    fn finish(mut self) -> DefMap {
        // Emit diagnostics for all remaining unexpanded macros.

        let _p = profile::span("DefCollector::finish");

        for directive in &self.unresolved_macros {
            match &directive.kind {
                MacroDirectiveKind::FnLike { ast_id, expand_to } => {
                    // FIXME: we shouldn't need to re-resolve the macro here just to get the unresolved error!
                    let macro_call_as_call_id = macro_call_as_call_id(
                        self.db.upcast(),
                        ast_id,
                        *expand_to,
                        self.def_map.krate,
                        |path| {
                            let resolved_res = self.def_map.resolve_path_fp_with_macro(
                                self.db,
                                ResolveMode::Other,
                                directive.module_id,
                                &path,
                                BuiltinShadowMode::Module,
                                Some(MacroSubNs::Bang),
                            );
                            resolved_res
                                .resolved_def
                                .take_macros()
                                .map(|it| macro_id_to_def_id(self.db, it))
                        },
                    );
                    if let Err(UnresolvedMacro { path }) = macro_call_as_call_id {
                        self.def_map.diagnostics.push(DefDiagnostic::unresolved_macro_call(
                            directive.module_id,
                            MacroCallKind::FnLike { ast_id: ast_id.ast_id, expand_to: *expand_to },
                            path,
                        ));
                    }
                }
                MacroDirectiveKind::Derive { ast_id, derive_attr, derive_pos } => {
                    self.def_map.diagnostics.push(DefDiagnostic::unresolved_macro_call(
                        directive.module_id,
                        MacroCallKind::Derive {
                            ast_id: ast_id.ast_id,
                            derive_attr_index: *derive_attr,
                            derive_index: *derive_pos as u32,
                        },
                        ast_id.path.clone(),
                    ));
                }
                // These are diagnosed by `reseed_with_unresolved_attribute`, as that function consumes them
                MacroDirectiveKind::Attr { .. } => {}
            }
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
                let item_tree = krate.item_tree(self.db);
                let extern_crate = &item_tree[krate.value];

                diagnosed_extern_crates.insert(extern_crate.name.clone());

                self.def_map.diagnostics.push(DefDiagnostic::unresolved_extern_crate(
                    directive.module_id,
                    InFile::new(krate.file_id(), extern_crate.ast_id),
                ));
            }
        }

        for directive in &self.unresolved_imports {
            if let ImportSource::Import { id: import, use_tree } = directive.import.source {
                if matches!(
                    (directive.import.path.segments().first(), &directive.import.path.kind),
                    (Some(krate), PathKind::Plain | PathKind::Abs) if diagnosed_extern_crates.contains(krate)
                ) {
                    continue;
                }

                self.def_map.diagnostics.push(DefDiagnostic::unresolved_import(
                    directive.module_id,
                    import,
                    use_tree,
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
    tree_id: TreeId,
    item_tree: &'a ItemTree,
    mod_dir: ModDir,
}

impl ModCollector<'_, '_> {
    fn collect_in_top_module(&mut self, items: &[ModItem]) {
        let module = self.def_collector.def_map.module_id(self.module_id);
        self.collect(items, module.into())
    }

    fn collect(&mut self, items: &[ModItem], container: ItemContainerId) {
        let krate = self.def_collector.def_map.krate;
        let is_crate_root = self.module_id == DefMap::ROOT;

        // Note: don't assert that inserted value is fresh: it's simply not true
        // for macros.
        self.def_collector.mod_dirs.insert(self.module_id, self.mod_dir.clone());

        // Prelude module is always considered to be `#[macro_use]`.
        if let Some(prelude_module) = self.def_collector.def_map.prelude {
            if prelude_module.krate != krate && is_crate_root {
                cov_mark::hit!(prelude_is_macro_use);
                self.def_collector.import_macros_from_extern_crate(prelude_module.krate, None);
            }
        }

        // This should be processed eagerly instead of deferred to resolving.
        // `#[macro_use] extern crate` is hoisted to imports macros before collecting
        // any other items.
        //
        // If we're not at the crate root, `macro_use`d extern crates are an error so let's just
        // ignore them.
        if is_crate_root {
            for &item in items {
                if let ModItem::ExternCrate(id) = item {
                    self.process_macro_use_extern_crate(id);
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

            if let Err(()) = self.resolve_attributes(&attrs, item, container) {
                // Do not process the item. It has at least one non-builtin attribute, so the
                // fixed-point algorithm is required to resolve the rest of them.
                continue;
            }

            let db = self.def_collector.db;
            let module = self.def_collector.def_map.module_id(self.module_id);
            let def_map = &mut self.def_collector.def_map;
            let update_def =
                |def_collector: &mut DefCollector<'_>, id, name: &Name, vis, has_constructor| {
                    def_collector.def_map.modules[self.module_id].scope.declare(id);
                    def_collector.update(
                        self.module_id,
                        &[(Some(name.clone()), PerNs::from_def(id, vis, has_constructor))],
                        vis,
                        ImportType::Named,
                    )
                };
            let resolve_vis = |def_map: &DefMap, visibility| {
                def_map
                    .resolve_visibility(db, self.module_id, visibility, false)
                    .unwrap_or(Visibility::Public)
            };

            match item {
                ModItem::Mod(m) => self.collect_module(m, &attrs),
                ModItem::Import(import_id) => {
                    let imports = Import::from_use(
                        db,
                        krate,
                        self.item_tree,
                        ItemTreeId::new(self.tree_id, import_id),
                    );
                    self.def_collector.unresolved_imports.extend(imports.into_iter().map(
                        |import| ImportDirective {
                            module_id: self.module_id,
                            import,
                            status: PartialResolvedImport::Unresolved,
                        },
                    ));
                }
                ModItem::ExternCrate(import_id) => {
                    self.def_collector.unresolved_imports.push(ImportDirective {
                        module_id: self.module_id,
                        import: Import::from_extern_crate(
                            db,
                            krate,
                            self.item_tree,
                            ItemTreeId::new(self.tree_id, import_id),
                        ),
                        status: PartialResolvedImport::Unresolved,
                    })
                }
                ModItem::ExternBlock(block) => self.collect(
                    &self.item_tree[block].children,
                    ItemContainerId::ExternBlockId(
                        ExternBlockLoc {
                            container: module,
                            id: ItemTreeId::new(self.tree_id, block),
                        }
                        .intern(db),
                    ),
                ),
                ModItem::MacroCall(mac) => self.collect_macro_call(&self.item_tree[mac], container),
                ModItem::MacroRules(id) => self.collect_macro_rules(id, module),
                ModItem::MacroDef(id) => self.collect_macro_def(id, module),
                ModItem::Impl(imp) => {
                    let impl_id =
                        ImplLoc { container: module, id: ItemTreeId::new(self.tree_id, imp) }
                            .intern(db);
                    self.def_collector.def_map.modules[self.module_id].scope.define_impl(impl_id)
                }
                ModItem::Function(id) => {
                    let it = &self.item_tree[id];
                    let fn_id =
                        FunctionLoc { container, id: ItemTreeId::new(self.tree_id, id) }.intern(db);

                    let vis = resolve_vis(def_map, &self.item_tree[it.visibility]);
                    if self.def_collector.is_proc_macro && self.module_id == DefMap::ROOT {
                        if let Some(proc_macro) = attrs.parse_proc_macro_decl(&it.name) {
                            self.def_collector.export_proc_macro(
                                proc_macro,
                                ItemTreeId::new(self.tree_id, id),
                                fn_id,
                            );
                        }
                    }

                    update_def(self.def_collector, fn_id.into(), &it.name, vis, false);
                }
                ModItem::Struct(id) => {
                    let it = &self.item_tree[id];

                    let vis = resolve_vis(def_map, &self.item_tree[it.visibility]);
                    update_def(
                        self.def_collector,
                        StructLoc { container: module, id: ItemTreeId::new(self.tree_id, id) }
                            .intern(db)
                            .into(),
                        &it.name,
                        vis,
                        !matches!(it.fields, Fields::Record(_)),
                    );
                }
                ModItem::Union(id) => {
                    let it = &self.item_tree[id];

                    let vis = resolve_vis(def_map, &self.item_tree[it.visibility]);
                    update_def(
                        self.def_collector,
                        UnionLoc { container: module, id: ItemTreeId::new(self.tree_id, id) }
                            .intern(db)
                            .into(),
                        &it.name,
                        vis,
                        false,
                    );
                }
                ModItem::Enum(id) => {
                    let it = &self.item_tree[id];

                    let vis = resolve_vis(def_map, &self.item_tree[it.visibility]);
                    update_def(
                        self.def_collector,
                        EnumLoc { container: module, id: ItemTreeId::new(self.tree_id, id) }
                            .intern(db)
                            .into(),
                        &it.name,
                        vis,
                        false,
                    );
                }
                ModItem::Const(id) => {
                    let it = &self.item_tree[id];
                    let const_id =
                        ConstLoc { container, id: ItemTreeId::new(self.tree_id, id) }.intern(db);

                    match &it.name {
                        Some(name) => {
                            let vis = resolve_vis(def_map, &self.item_tree[it.visibility]);
                            update_def(self.def_collector, const_id.into(), name, vis, false);
                        }
                        None => {
                            // const _: T = ...;
                            self.def_collector.def_map.modules[self.module_id]
                                .scope
                                .define_unnamed_const(const_id);
                        }
                    }
                }
                ModItem::Static(id) => {
                    let it = &self.item_tree[id];

                    let vis = resolve_vis(def_map, &self.item_tree[it.visibility]);
                    update_def(
                        self.def_collector,
                        StaticLoc { container, id: ItemTreeId::new(self.tree_id, id) }
                            .intern(db)
                            .into(),
                        &it.name,
                        vis,
                        false,
                    );
                }
                ModItem::Trait(id) => {
                    let it = &self.item_tree[id];

                    let vis = resolve_vis(def_map, &self.item_tree[it.visibility]);
                    update_def(
                        self.def_collector,
                        TraitLoc { container: module, id: ItemTreeId::new(self.tree_id, id) }
                            .intern(db)
                            .into(),
                        &it.name,
                        vis,
                        false,
                    );
                }
                ModItem::TraitAlias(id) => {
                    let it = &self.item_tree[id];

                    let vis = resolve_vis(def_map, &self.item_tree[it.visibility]);
                    update_def(
                        self.def_collector,
                        TraitAliasLoc { container: module, id: ItemTreeId::new(self.tree_id, id) }
                            .intern(db)
                            .into(),
                        &it.name,
                        vis,
                        false,
                    );
                }
                ModItem::TypeAlias(id) => {
                    let it = &self.item_tree[id];

                    let vis = resolve_vis(def_map, &self.item_tree[it.visibility]);
                    update_def(
                        self.def_collector,
                        TypeAliasLoc { container, id: ItemTreeId::new(self.tree_id, id) }
                            .intern(db)
                            .into(),
                        &it.name,
                        vis,
                        false,
                    );
                }
            }
        }
    }

    fn process_macro_use_extern_crate(&mut self, extern_crate: FileItemTreeId<ExternCrate>) {
        let db = self.def_collector.db;
        let attrs = self.item_tree.attrs(
            db,
            self.def_collector.def_map.krate,
            ModItem::from(extern_crate).into(),
        );
        if let Some(cfg) = attrs.cfg() {
            if !self.is_cfg_enabled(&cfg) {
                return;
            }
        }

        let target_crate =
            match self.def_collector.resolve_extern_crate(&self.item_tree[extern_crate].name) {
                Some(m) if m.krate == self.def_collector.def_map.krate => {
                    cov_mark::hit!(ignore_macro_use_extern_crate_self);
                    return;
                }
                Some(m) => m.krate,
                None => return,
            };

        cov_mark::hit!(macro_rules_from_other_crates_are_visible_with_macro_use);

        let mut single_imports = Vec::new();
        let hygiene = Hygiene::new_unhygienic();
        for attr in attrs.by_key("macro_use").attrs() {
            let Some(paths) = attr.parse_path_comma_token_tree(db.upcast(), &hygiene) else {
                // `#[macro_use]` (without any paths) found, forget collected names and just import
                // all visible macros.
                self.def_collector.import_macros_from_extern_crate(target_crate, None);
                return;
            };
            for path in paths {
                if let Some(name) = path.as_ident() {
                    single_imports.push(name.clone());
                }
            }
        }

        self.def_collector.import_macros_from_extern_crate(target_crate, Some(single_imports));
    }

    fn collect_module(&mut self, module_id: FileItemTreeId<Mod>, attrs: &Attrs) {
        let path_attr = attrs.by_key("path").string_value();
        let is_macro_use = attrs.by_key("macro_use").exists();
        let module = &self.item_tree[module_id];
        match &module.kind {
            // inline module, just recurse
            ModKind::Inline { items } => {
                let module_id = self.push_child_module(
                    module.name.clone(),
                    AstId::new(self.file_id(), module.ast_id),
                    None,
                    &self.item_tree[module.visibility],
                    module_id,
                );

                if let Some(mod_dir) = self.mod_dir.descend_into_definition(&module.name, path_attr)
                {
                    ModCollector {
                        def_collector: &mut *self.def_collector,
                        macro_depth: self.macro_depth,
                        module_id,
                        tree_id: self.tree_id,
                        item_tree: self.item_tree,
                        mod_dir,
                    }
                    .collect_in_top_module(&*items);
                    if is_macro_use {
                        self.import_all_legacy_macros(module_id);
                    }
                }
            }
            // out of line module, resolve, parse and recurse
            ModKind::Outline => {
                let ast_id = AstId::new(self.tree_id.file_id(), module.ast_id);
                let db = self.def_collector.db;
                match self.mod_dir.resolve_declaration(db, self.file_id(), &module.name, path_attr)
                {
                    Ok((file_id, is_mod_rs, mod_dir)) => {
                        let item_tree = db.file_item_tree(file_id.into());
                        let krate = self.def_collector.def_map.krate;
                        let is_enabled = item_tree
                            .top_level_attrs(db, krate)
                            .cfg()
                            .map_or(true, |cfg| self.is_cfg_enabled(&cfg));
                        if is_enabled {
                            let module_id = self.push_child_module(
                                module.name.clone(),
                                ast_id,
                                Some((file_id, is_mod_rs)),
                                &self.item_tree[module.visibility],
                                module_id,
                            );
                            ModCollector {
                                def_collector: self.def_collector,
                                macro_depth: self.macro_depth,
                                module_id,
                                tree_id: TreeId::new(file_id.into(), None),
                                item_tree: &item_tree,
                                mod_dir,
                            }
                            .collect_in_top_module(item_tree.top_level_items());
                            let is_macro_use = is_macro_use
                                || item_tree
                                    .top_level_attrs(db, krate)
                                    .by_key("macro_use")
                                    .exists();
                            if is_macro_use {
                                self.import_all_legacy_macros(module_id);
                            }
                        }
                    }
                    Err(candidates) => {
                        self.push_child_module(
                            module.name.clone(),
                            ast_id,
                            None,
                            &self.item_tree[module.visibility],
                            module_id,
                        );
                        self.def_collector.def_map.diagnostics.push(
                            DefDiagnostic::unresolved_module(self.module_id, ast_id, candidates),
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
        mod_tree_id: FileItemTreeId<Mod>,
    ) -> LocalModuleId {
        let def_map = &mut self.def_collector.def_map;
        let vis = def_map
            .resolve_visibility(self.def_collector.db, self.module_id, visibility, false)
            .unwrap_or(Visibility::Public);
        let origin = match definition {
            None => ModuleOrigin::Inline {
                definition: declaration,
                definition_tree_id: ItemTreeId::new(self.tree_id, mod_tree_id),
            },
            Some((definition, is_mod_rs)) => ModuleOrigin::File {
                declaration,
                definition,
                is_mod_rs,
                declaration_tree_id: ItemTreeId::new(self.tree_id, mod_tree_id),
            },
        };

        let modules = &mut def_map.modules;
        let res = modules.alloc(ModuleData::new(origin, vis));
        modules[res].parent = Some(self.module_id);

        if let Some((target, source)) = Self::borrow_modules(modules.as_mut(), res, self.module_id)
        {
            for (name, macs) in source.scope.legacy_macros() {
                for &mac in macs {
                    target.scope.define_legacy_macro(name.clone(), mac);
                }
            }
        }
        modules[self.module_id].children.insert(name.clone(), res);

        let module = def_map.module_id(res);
        let def = ModuleDefId::from(module);

        def_map.modules[self.module_id].scope.declare(def);
        self.def_collector.update(
            self.module_id,
            &[(Some(name), PerNs::from_def(def, vis, false))],
            vis,
            ImportType::Named,
        );
        res
    }

    /// Resolves attributes on an item.
    ///
    /// Returns `Err` when some attributes could not be resolved to builtins and have been
    /// registered as unresolved.
    ///
    /// If `ignore_up_to` is `Some`, attributes preceding and including that attribute will be
    /// assumed to be resolved already.
    fn resolve_attributes(
        &mut self,
        attrs: &Attrs,
        mod_item: ModItem,
        container: ItemContainerId,
    ) -> Result<(), ()> {
        let mut ignore_up_to =
            self.def_collector.skip_attrs.get(&InFile::new(self.file_id(), mod_item)).copied();
        let iter = attrs
            .iter()
            .dedup_by(|a, b| {
                // FIXME: this should not be required, all attributes on an item should have a
                // unique ID!
                // Still, this occurs because `#[cfg_attr]` can "expand" to multiple attributes:
                //     #[cfg_attr(not(off), unresolved, unresolved)]
                //     struct S;
                // We should come up with a different way to ID attributes.
                a.id == b.id
            })
            .skip_while(|attr| match ignore_up_to {
                Some(id) if attr.id == id => {
                    ignore_up_to = None;
                    true
                }
                Some(_) => true,
                None => false,
            });

        for attr in iter {
            if self.def_collector.def_map.is_builtin_or_registered_attr(&attr.path) {
                continue;
            }
            tracing::debug!(
                "non-builtin attribute {}",
                attr.path.display(self.def_collector.db.upcast())
            );

            let ast_id = AstIdWithPath::new(
                self.file_id(),
                mod_item.ast_id(self.item_tree),
                attr.path.as_ref().clone(),
            );
            self.def_collector.unresolved_macros.push(MacroDirective {
                module_id: self.module_id,
                depth: self.macro_depth + 1,
                kind: MacroDirectiveKind::Attr {
                    ast_id,
                    attr: attr.clone(),
                    mod_item,
                    tree: self.tree_id,
                },
                container,
            });

            return Err(());
        }

        Ok(())
    }

    fn collect_macro_rules(&mut self, id: FileItemTreeId<MacroRules>, module: ModuleId) {
        let krate = self.def_collector.def_map.krate;
        let mac = &self.item_tree[id];
        let attrs = self.item_tree.attrs(self.def_collector.db, krate, ModItem::from(id).into());
        let ast_id = InFile::new(self.file_id(), mac.ast_id.upcast());

        let export_attr = attrs.by_key("macro_export");

        let is_export = export_attr.exists();
        let local_inner = if is_export {
            export_attr.tt_values().flat_map(|it| &it.token_trees).any(|it| match it {
                tt::TokenTree::Leaf(tt::Leaf::Ident(ident)) => {
                    ident.text.contains("local_inner_macros")
                }
                _ => false,
            })
        } else {
            false
        };

        // Case 1: builtin macros
        let expander = if attrs.by_key("rustc_builtin_macro").exists() {
            // `#[rustc_builtin_macro = "builtin_name"]` overrides the `macro_rules!` name.
            let name;
            let name = match attrs.by_key("rustc_builtin_macro").string_value() {
                Some(it) => {
                    // FIXME: a hacky way to create a Name from string.
                    name =
                        tt::Ident { text: it.clone(), span: tt::TokenId::unspecified() }.as_name();
                    &name
                }
                None => {
                    let explicit_name =
                        attrs.by_key("rustc_builtin_macro").tt_values().next().and_then(|tt| {
                            match tt.token_trees.first() {
                                Some(tt::TokenTree::Leaf(tt::Leaf::Ident(name))) => Some(name),
                                _ => None,
                            }
                        });
                    match explicit_name {
                        Some(ident) => {
                            name = ident.as_name();
                            &name
                        }
                        None => &mac.name,
                    }
                }
            };
            match find_builtin_macro(name) {
                Some(Either::Left(it)) => MacroExpander::BuiltIn(it),
                Some(Either::Right(it)) => MacroExpander::BuiltInEager(it),
                None => {
                    self.def_collector
                        .def_map
                        .diagnostics
                        .push(DefDiagnostic::unimplemented_builtin_macro(self.module_id, ast_id));
                    return;
                }
            }
        } else {
            // Case 2: normal `macro_rules!` macro
            MacroExpander::Declarative
        };
        let allow_internal_unsafe = attrs.by_key("allow_internal_unsafe").exists();

        let macro_id = MacroRulesLoc {
            container: module,
            id: ItemTreeId::new(self.tree_id, id),
            local_inner,
            allow_internal_unsafe,
            expander,
        }
        .intern(self.def_collector.db);
        self.def_collector.define_macro_rules(
            self.module_id,
            mac.name.clone(),
            macro_id,
            is_export,
        );
    }

    fn collect_macro_def(&mut self, id: FileItemTreeId<MacroDef>, module: ModuleId) {
        let krate = self.def_collector.def_map.krate;
        let mac = &self.item_tree[id];
        let ast_id = InFile::new(self.file_id(), mac.ast_id.upcast());

        // Case 1: builtin macros
        let mut helpers_opt = None;
        let attrs = self.item_tree.attrs(self.def_collector.db, krate, ModItem::from(id).into());
        let expander = if attrs.by_key("rustc_builtin_macro").exists() {
            if let Some(expander) = find_builtin_macro(&mac.name) {
                match expander {
                    Either::Left(it) => MacroExpander::BuiltIn(it),
                    Either::Right(it) => MacroExpander::BuiltInEager(it),
                }
            } else if let Some(expander) = find_builtin_derive(&mac.name) {
                if let Some(attr) = attrs.by_key("rustc_builtin_macro").tt_values().next() {
                    // NOTE: The item *may* have both `#[rustc_builtin_macro]` and `#[proc_macro_derive]`,
                    // in which case rustc ignores the helper attributes from the latter, but it
                    // "doesn't make sense in practice" (see rust-lang/rust#87027).
                    if let Some((name, helpers)) =
                        parse_macro_name_and_helper_attrs(&attr.token_trees)
                    {
                        // NOTE: rustc overrides the name if the macro name if it's different from the
                        // macro name, but we assume it isn't as there's no such case yet. FIXME if
                        // the following assertion fails.
                        stdx::always!(
                            name == mac.name,
                            "built-in macro {} has #[rustc_builtin_macro] which declares different name {}",
                            mac.name.display(self.def_collector.db.upcast()),
                            name.display(self.def_collector.db.upcast())
                        );
                        helpers_opt = Some(helpers);
                    }
                }
                MacroExpander::BuiltInDerive(expander)
            } else if let Some(expander) = find_builtin_attr(&mac.name) {
                MacroExpander::BuiltInAttr(expander)
            } else {
                self.def_collector
                    .def_map
                    .diagnostics
                    .push(DefDiagnostic::unimplemented_builtin_macro(self.module_id, ast_id));
                return;
            }
        } else {
            // Case 2: normal `macro`
            MacroExpander::Declarative
        };
        let allow_internal_unsafe = attrs.by_key("allow_internal_unsafe").exists();

        let macro_id = Macro2Loc {
            container: module,
            id: ItemTreeId::new(self.tree_id, id),
            expander,
            allow_internal_unsafe,
        }
        .intern(self.def_collector.db);
        self.def_collector.define_macro_def(
            self.module_id,
            mac.name.clone(),
            macro_id,
            &self.item_tree[mac.visibility],
        );
        if let Some(helpers) = helpers_opt {
            if self.def_collector.def_map.block.is_none() {
                Arc::get_mut(&mut self.def_collector.def_map.data)
                    .unwrap()
                    .exported_derives
                    .insert(macro_id_to_def_id(self.def_collector.db, macro_id.into()), helpers);
            }
        }
    }

    fn collect_macro_call(&mut self, mac: &MacroCall, container: ItemContainerId) {
        let ast_id = AstIdWithPath::new(self.file_id(), mac.ast_id, ModPath::clone(&mac.path));
        let db = self.def_collector.db;

        // FIXME: Immediately expanding in "Case 1" is insufficient since "Case 2" may also define
        // new legacy macros that create textual scopes. We need a way to resolve names in textual
        // scopes without eager expansion.

        // Case 1: try to resolve macro calls with single-segment name and expand macro_rules
        if let Ok(res) = macro_call_as_call_id(
            db.upcast(),
            &ast_id,
            mac.expand_to,
            self.def_collector.def_map.krate,
            |path| {
                path.as_ident().and_then(|name| {
                    let def_map = &self.def_collector.def_map;
                    def_map
                        .with_ancestor_maps(db, self.module_id, &mut |map, module| {
                            map[module].scope.get_legacy_macro(name)?.last().copied()
                        })
                        .or_else(|| def_map[self.module_id].scope.get(name).take_macros())
                        .or_else(|| def_map.macro_use_prelude.get(name).copied())
                        .filter(|&id| {
                            sub_namespace_match(
                                Some(MacroSubNs::from_id(db, id)),
                                Some(MacroSubNs::Bang),
                            )
                        })
                        .map(|it| macro_id_to_def_id(self.def_collector.db, it))
                })
            },
        ) {
            // Legacy macros need to be expanded immediately, so that any macros they produce
            // are in scope.
            if let Some(val) = res {
                self.def_collector.collect_macro_expansion(
                    self.module_id,
                    val,
                    self.macro_depth + 1,
                    container,
                );
            }

            return;
        }

        // Case 2: resolve in module scope, expand during name resolution.
        self.def_collector.unresolved_macros.push(MacroDirective {
            module_id: self.module_id,
            depth: self.macro_depth + 1,
            kind: MacroDirectiveKind::FnLike { ast_id, expand_to: mac.expand_to },
            container,
        });
    }

    fn import_all_legacy_macros(&mut self, module_id: LocalModuleId) {
        let Some((source, target)) = Self::borrow_modules(self.def_collector.def_map.modules.as_mut(), module_id, self.module_id) else {
            return
        };

        for (name, macs) in source.scope.legacy_macros() {
            macs.last().map(|&mac| {
                target.scope.define_legacy_macro(name.clone(), mac);
            });
        }
    }

    /// Mutably borrow two modules at once, retu
    fn borrow_modules(
        modules: &mut [ModuleData],
        a: LocalModuleId,
        b: LocalModuleId,
    ) -> Option<(&mut ModuleData, &mut ModuleData)> {
        let a = a.into_raw().into_u32() as usize;
        let b = b.into_raw().into_u32() as usize;

        let (a, b) = match a.cmp(&b) {
            Ordering::Equal => return None,
            Ordering::Less => {
                let (prefix, b) = modules.split_at_mut(b);
                (&mut prefix[a], &mut b[0])
            }
            Ordering::Greater => {
                let (prefix, a) = modules.split_at_mut(a);
                (&mut a[0], &mut prefix[b])
            }
        };
        Some((a, b))
    }

    fn is_cfg_enabled(&self, cfg: &CfgExpr) -> bool {
        self.def_collector.cfg_options.check(cfg) != Some(false)
    }

    fn emit_unconfigured_diagnostic(&mut self, item: ModItem, cfg: &CfgExpr) {
        let ast_id = item.ast_id(self.item_tree);

        let ast_id = InFile::new(self.file_id(), ast_id.upcast());
        self.def_collector.def_map.diagnostics.push(DefDiagnostic::unconfigured_code(
            self.module_id,
            ast_id,
            cfg.clone(),
            self.def_collector.cfg_options.clone(),
        ));
    }

    fn file_id(&self) -> HirFileId {
        self.tree_id.file_id()
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
            deps: FxHashMap::default(),
            glob_imports: FxHashMap::default(),
            unresolved_imports: Vec::new(),
            indeterminate_imports: Vec::new(),
            unresolved_macros: Vec::new(),
            mod_dirs: FxHashMap::default(),
            cfg_options: &CfgOptions::default(),
            proc_macros: Ok(vec![]),
            from_glob_import: Default::default(),
            skip_attrs: Default::default(),
            is_proc_macro: false,
            hygienes: FxHashMap::default(),
        };
        collector.seed_with_top_level();
        collector.collect();
        collector.def_map
    }

    fn do_resolve(not_ra_fixture: &str) -> DefMap {
        let (db, file_id) = TestDB::with_single_file(not_ra_fixture);
        let krate = db.test_crate();

        let edition = db.crate_graph()[krate].edition;
        let module_origin = ModuleOrigin::CrateRoot { definition: file_id };
        let def_map =
            DefMap::empty(krate, edition, ModuleData::new(module_origin, Visibility::Public));
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
        do_resolve(
            r#"
macro_rules! foo {
    ($($ty:ty)*) => { foo!(() $($ty)*); }
}
foo!(KABOOM);
"#,
        );
    }

    #[ignore]
    #[test]
    fn test_macro_expand_will_stop_2() {
        // FIXME: this test does succeed, but takes quite a while: 90 seconds in
        // the release mode. That's why the argument is not an ra_fixture --
        // otherwise injection highlighting gets stuck.
        //
        // We need to find a way to fail this faster.
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
