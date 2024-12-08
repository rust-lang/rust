//! The core of the module-level name resolution algorithm.
//!
//! `DefCollector::collect` contains the fixed-point iteration loop which
//! resolves imports and expands macros.

use std::{cmp::Ordering, iter, mem, ops::Not};

use base_db::{CrateId, CrateOrigin, Dependency, LangCrateOrigin};
use cfg::{CfgAtom, CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{
    attrs::{Attr, AttrId},
    builtin::{find_builtin_attr, find_builtin_derive, find_builtin_macro},
    name::{AsName, Name},
    proc_macro::CustomProcMacroExpander,
    ExpandTo, HirFileId, InFile, MacroCallId, MacroCallKind, MacroDefId, MacroDefKind,
    MacroFileIdExt,
};
use intern::{sym, Interned};
use itertools::{izip, Itertools};
use la_arena::Idx;
use limit::Limit;
use rustc_hash::{FxHashMap, FxHashSet};
use span::{Edition, EditionedFileId, FileAstId, SyntaxContextId};
use syntax::ast;
use triomphe::Arc;

use crate::{
    attr::Attrs,
    db::DefDatabase,
    item_scope::{ImportId, ImportOrExternCrate, ImportType, PerNsGlobImports},
    item_tree::{
        self, AttrOwner, FieldsShape, FileItemTreeId, ImportKind, ItemTree, ItemTreeId,
        ItemTreeNode, Macro2, MacroCall, MacroRules, Mod, ModItem, ModKind, TreeId, UseTreeKind,
    },
    macro_call_as_call_id, macro_call_as_call_id_with_eager,
    nameres::{
        attr_resolution::{attr_macro_as_call_id, derive_macro_as_call_id},
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
    AdtId, AstId, AstIdWithPath, ConstLoc, CrateRootModuleId, EnumLoc, EnumVariantLoc,
    ExternBlockLoc, ExternCrateId, ExternCrateLoc, FunctionId, FunctionLoc, ImplLoc, Intern,
    ItemContainerId, LocalModuleId, Lookup, Macro2Id, Macro2Loc, MacroExpander, MacroId,
    MacroRulesId, MacroRulesLoc, MacroRulesLocFlags, ModuleDefId, ModuleId, ProcMacroId,
    ProcMacroLoc, StaticLoc, StructLoc, TraitAliasLoc, TraitLoc, TypeAliasLoc, UnionLoc,
    UnresolvedMacro, UseId, UseLoc,
};

static GLOB_RECURSION_LIMIT: Limit = Limit::new(100);
static FIXED_POINT_LIMIT: Limit = Limit::new(8192);

pub(super) fn collect_defs(db: &dyn DefDatabase, def_map: DefMap, tree_id: TreeId) -> DefMap {
    let crate_graph = db.crate_graph();

    let krate = &crate_graph[def_map.krate];

    // populate external prelude and dependency list
    let mut deps =
        FxHashMap::with_capacity_and_hasher(krate.dependencies.len(), Default::default());
    for dep in &krate.dependencies {
        tracing::debug!("crate dep {:?} -> {:?}", dep.name, dep.crate_id);

        deps.insert(dep.as_name(), dep.clone());
    }

    let proc_macros = if krate.is_proc_macro {
        db.proc_macros()
            .for_crate(def_map.krate, db.syntax_context(tree_id.file_id()))
            .unwrap_or_default()
    } else {
        Default::default()
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
        cfg_options: &krate.cfg_options,
        proc_macros,
        from_glob_import: Default::default(),
        skip_attrs: Default::default(),
        unresolved_extern_crates: Default::default(),
        is_proc_macro: krate.is_proc_macro,
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
struct ImportSource {
    use_tree: Idx<ast::UseTree>,
    id: UseId,
    is_prelude: bool,
    kind: ImportKind,
}

#[derive(Debug, Eq, PartialEq)]
struct Import {
    path: ModPath,
    alias: Option<ImportAlias>,
    visibility: RawVisibility,
    source: ImportSource,
}

impl Import {
    fn from_use(
        tree: &ItemTree,
        item_tree_id: ItemTreeId<item_tree::Use>,
        id: UseId,
        is_prelude: bool,
        mut cb: impl FnMut(Self),
    ) {
        let it = &tree[item_tree_id.value];
        let visibility = &tree[it.visibility];
        it.use_tree.expand(|idx, path, kind, alias| {
            cb(Self {
                path,
                alias,
                visibility: visibility.clone(),
                source: ImportSource { use_tree: idx, id, is_prelude, kind },
            });
        });
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
    FnLike {
        ast_id: AstIdWithPath<ast::MacroCall>,
        expand_to: ExpandTo,
        ctxt: SyntaxContextId,
    },
    Derive {
        ast_id: AstIdWithPath<ast::Adt>,
        derive_attr: AttrId,
        derive_pos: usize,
        ctxt: SyntaxContextId,
        /// The "parent" macro it is resolved to.
        derive_macro_id: MacroCallId,
    },
    Attr {
        ast_id: AstIdWithPath<ast::Item>,
        attr: Attr,
        mod_item: ModItem,
        /* is this needed? */ tree: TreeId,
    },
}

/// Walks the tree of module recursively
struct DefCollector<'a> {
    db: &'a dyn DefDatabase,
    def_map: DefMap,
    // The dependencies of the current crate, including optional deps like `test`.
    deps: FxHashMap<Name, Dependency>,
    glob_imports: FxHashMap<LocalModuleId, Vec<(LocalModuleId, Visibility, UseId)>>,
    unresolved_imports: Vec<ImportDirective>,
    indeterminate_imports: Vec<(ImportDirective, PerNs)>,
    unresolved_macros: Vec<MacroDirective>,
    // We'd like to avoid emitting a diagnostics avalanche when some `extern crate` doesn't
    // resolve. When we emit diagnostics for unresolved imports, we only do so if the import
    // doesn't start with an unresolved crate's name.
    unresolved_extern_crates: FxHashSet<Name>,
    mod_dirs: FxHashMap<LocalModuleId, ModDir>,
    cfg_options: &'a CfgOptions,
    /// List of procedural macros defined by this crate. This is read from the dynamic library
    /// built by the build system, and is the list of proc-macros we can actually expand. It is
    /// empty when proc-macro support is disabled (in which case we still do name resolution for
    /// them). The bool signals whether the proc-macro has been explicitly disabled for name-resolution.
    proc_macros: Box<[(Name, CustomProcMacroExpander, bool)]>,
    is_proc_macro: bool,
    from_glob_import: PerNsGlobImports,
    /// If we fail to resolve an attribute on a `ModItem`, we fall back to ignoring the attribute.
    /// This map is used to skip all attributes up to and including the one that failed to resolve,
    /// in order to not expand them twice.
    ///
    /// This also stores the attributes to skip when we resolve derive helpers and non-macro
    /// non-builtin attributes in general.
    // FIXME: There has to be a better way to do this
    skip_attrs: FxHashMap<InFile<ModItem>, AttrId>,
}

impl DefCollector<'_> {
    fn seed_with_top_level(&mut self) {
        let _p = tracing::info_span!("seed_with_top_level").entered();

        let crate_graph = self.db.crate_graph();
        let file_id = crate_graph[self.def_map.krate].root_file_id();
        let item_tree = self.db.file_item_tree(file_id.into());
        let attrs = item_tree.top_level_attrs(self.db, self.def_map.krate);
        let crate_data = Arc::get_mut(&mut self.def_map.data).unwrap();

        let mut process = true;

        // Process other crate-level attributes.
        for attr in &*attrs {
            if let Some(cfg) = attr.cfg() {
                if self.cfg_options.check(&cfg) == Some(false) {
                    process = false;
                    break;
                }
            }
            let Some(attr_name) = attr.path.as_ident() else { continue };

            match () {
                () if *attr_name == sym::recursion_limit.clone() => {
                    if let Some(limit) = attr.string_value() {
                        if let Ok(limit) = limit.as_str().parse() {
                            crate_data.recursion_limit = Some(limit);
                        }
                    }
                }
                () if *attr_name == sym::crate_type.clone() => {
                    if attr.string_value() == Some(&sym::proc_dash_macro) {
                        self.is_proc_macro = true;
                    }
                }
                () if *attr_name == sym::no_core.clone() => crate_data.no_core = true,
                () if *attr_name == sym::no_std.clone() => crate_data.no_std = true,
                () if *attr_name == sym::rustc_coherence_is_core.clone() => {
                    crate_data.rustc_coherence_is_core = true;
                }
                () if *attr_name == sym::feature.clone() => {
                    let features = attr
                        .parse_path_comma_token_tree(self.db.upcast())
                        .into_iter()
                        .flatten()
                        .filter_map(|(feat, _)| match feat.segments() {
                            [name] => Some(name.symbol().clone()),
                            _ => None,
                        });
                    crate_data.unstable_features.extend(features);
                }
                () if *attr_name == sym::register_attr.clone() => {
                    if let Some(ident) = attr.single_ident_value() {
                        crate_data.registered_attrs.push(ident.sym.clone());
                        cov_mark::hit!(register_attr);
                    }
                }
                () if *attr_name == sym::register_tool.clone() => {
                    if let Some(ident) = attr.single_ident_value() {
                        crate_data.registered_tools.push(ident.sym.clone());
                        cov_mark::hit!(register_tool);
                    }
                }
                () => (),
            }
        }

        for (name, dep) in &self.deps {
            // Add all
            if dep.is_prelude() {
                // This is a bit confusing but the gist is that `no_core` and `no_std` remove the
                // sysroot dependence on `core` and `std` respectively. Our `CrateGraph` is eagerly
                // constructed with them in place no matter what though, since at that point we
                // don't do pre-configured attribute resolution yet.
                // So here check if we are no_core / no_std and we are trying to add the
                // corresponding dep from the sysroot
                let skip = match crate_graph[dep.crate_id].origin {
                    CrateOrigin::Lang(LangCrateOrigin::Core) => {
                        crate_data.no_core && dep.is_sysroot()
                    }
                    CrateOrigin::Lang(LangCrateOrigin::Std) => {
                        crate_data.no_std && dep.is_sysroot()
                    }
                    _ => false,
                };
                if skip {
                    continue;
                }

                crate_data
                    .extern_prelude
                    .insert(name.clone(), (CrateRootModuleId { krate: dep.crate_id }, None));
            }
        }

        self.inject_prelude();

        if !process {
            return;
        }

        ModCollector {
            def_collector: self,
            macro_depth: 0,
            module_id: DefMap::ROOT,
            tree_id: TreeId::new(file_id.into(), None),
            item_tree: &item_tree,
            mod_dir: ModDir::root(),
        }
        .collect_in_top_module(item_tree.top_level_items());
        Arc::get_mut(&mut self.def_map.data).unwrap().shrink_to_fit();
    }

    fn seed_with_inner(&mut self, tree_id: TreeId) {
        let item_tree = tree_id.item_tree(self.db);
        let is_cfg_enabled = item_tree
            .top_level_attrs(self.db, self.def_map.krate)
            .cfg()
            .map_or(true, |cfg| self.cfg_options.check(&cfg) != Some(false));
        if is_cfg_enabled {
            self.inject_prelude();

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
        let _p = tracing::info_span!("DefCollector::resolution_loop").entered();

        // main name resolution fixed-point loop.
        let mut i = 0;
        'resolve_attr: loop {
            let _p = tracing::info_span!("resolve_macros loop").entered();
            'resolve_macros: loop {
                self.db.unwind_if_cancelled();

                {
                    let _p = tracing::info_span!("resolve_imports loop").entered();

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
        let _p = tracing::info_span!("DefCollector::collect").entered();

        self.resolution_loop();

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
            let module_id = self.def_map.module_id(DefMap::ROOT);
            let root = &mut self.def_map.modules[DefMap::ROOT];
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
                            attr_args: None,
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
            Name::new_symbol_root(sym::core.clone())
        } else if self.def_map.extern_prelude().any(|(name, _)| *name == sym::std.clone()) {
            Name::new_symbol_root(sym::std.clone())
        } else {
            // If `std` does not exist for some reason, fall back to core. This mostly helps
            // keep r-a's own tests minimal.
            Name::new_symbol_root(sym::core.clone())
        };

        let edition = match self.def_map.data.edition {
            Edition::Edition2015 => Name::new_symbol_root(sym::rust_2015.clone()),
            Edition::Edition2018 => Name::new_symbol_root(sym::rust_2018.clone()),
            Edition::Edition2021 => Name::new_symbol_root(sym::rust_2021.clone()),
            Edition::Edition2024 => Name::new_symbol_root(sym::rust_2024.clone()),
        };

        let path_kind = match self.def_map.data.edition {
            Edition::Edition2015 => PathKind::Plain,
            _ => PathKind::Abs,
        };
        let path = ModPath::from_segments(
            path_kind,
            [krate, Name::new_symbol_root(sym::prelude.clone()), edition],
        );

        let (per_ns, _) =
            self.def_map.resolve_path(self.db, DefMap::ROOT, &path, BuiltinShadowMode::Other, None);

        match per_ns.types {
            Some((ModuleDefId::ModuleId(m), _, import)) => {
                // FIXME: This should specifically look for a glob import somehow and record that here
                self.def_map.prelude = Some((
                    m,
                    import.and_then(ImportOrExternCrate::into_import).map(|it| it.import),
                ));
            }
            types => {
                tracing::debug!(
                    "could not resolve prelude path `{}` to module (resolved to {:?})",
                    path.display(self.db.upcast(), Edition::LATEST),
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
        let kind = def.kind.to_basedb_kind();
        let (expander, kind) = match self.proc_macros.iter().find(|(n, _, _)| n == &def.name) {
            Some(_)
                if kind == hir_expand::proc_macro::ProcMacroKind::Attr
                    && !self.db.expand_proc_attr_macros() =>
            {
                (CustomProcMacroExpander::disabled_proc_attr(), kind)
            }
            Some(&(_, _, true)) => (CustomProcMacroExpander::disabled(), kind),
            Some(&(_, expander, false)) => (expander, kind),
            None => (CustomProcMacroExpander::missing_expander(), kind),
        };

        let proc_macro_id = ProcMacroLoc {
            container: self.def_map.crate_root(),
            id,
            expander,
            kind,
            edition: self.def_map.data.edition,
        }
        .intern(self.db);
        self.define_proc_macro(def.name.clone(), proc_macro_id);
        let crate_data = Arc::get_mut(&mut self.def_map.data).unwrap();
        if let ProcMacroKind::Derive { helpers } = def.kind {
            crate_data.exported_derives.insert(self.db.macro_def(proc_macro_id.into()), helpers);
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
                &[(Some(name), PerNs::macros(macro_.into(), Visibility::Public, None))],
                Visibility::Public,
                None,
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
            &[(Some(name), PerNs::macros(macro_.into(), Visibility::Public, None))],
            vis,
            None,
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
            &[(Some(name), PerNs::macros(macro_.into(), Visibility::Public, None))],
            Visibility::Public,
            None,
        );
    }

    /// Import exported macros from another crate. `names`, if `Some(_)`, specifies the name of
    /// macros to be imported. Otherwise this method imports all exported macros.
    ///
    /// Exported macros are just all macros in the root module scope.
    /// Note that it contains not only all `#[macro_export]` macros, but also all aliases
    /// created by `use` in the root module, ignoring the visibility of `use`.
    fn import_macros_from_extern_crate(
        &mut self,
        krate: CrateId,
        names: Option<Vec<Name>>,
        extern_crate: Option<ExternCrateId>,
    ) {
        let def_map = self.db.crate_def_map(krate);
        // `#[macro_use]` brings macros into macro_use prelude. Yes, even non-`macro_rules!`
        // macros.
        let root_scope = &def_map[DefMap::ROOT].scope;
        match names {
            Some(names) => {
                for name in names {
                    // FIXME: Report diagnostic on 404.
                    if let Some(def) = root_scope.get(&name).take_macros() {
                        self.def_map.macro_use_prelude.insert(name, (def, extern_crate));
                    }
                }
            }
            None => {
                for (name, def) in root_scope.macros() {
                    self.def_map.macro_use_prelude.insert(name.clone(), (def, extern_crate));
                }
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
                    PartialResolvedImport::Indeterminate(resolved) => {
                        self.record_resolved_import(&directive);
                        self.indeterminate_imports.push((directive, resolved));
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

        // Resolve all indeterminate resolved imports again
        // As some of the macros will expand newly import shadowing partial resolved imports
        // FIXME: We maybe could skip this, if we handle the indeterminate imports in `resolve_imports`
        // correctly
        let mut indeterminate_imports = std::mem::take(&mut self.indeterminate_imports);
        indeterminate_imports.retain_mut(|(directive, partially_resolved)| {
            let partially_resolved = partially_resolved.availability();
            directive.status = self.resolve_import(directive.module_id, &directive.import);
            match directive.status {
                PartialResolvedImport::Indeterminate(import)
                    if partially_resolved != import.availability() =>
                {
                    self.record_resolved_import(directive);
                    res = ReachedFixedPoint::No;
                    false
                }
                PartialResolvedImport::Resolved(_) => {
                    self.record_resolved_import(directive);
                    res = ReachedFixedPoint::No;
                    false
                }
                _ => true,
            }
        });
        self.indeterminate_imports = indeterminate_imports;

        res
    }

    fn resolve_import(&self, module_id: LocalModuleId, import: &Import) -> PartialResolvedImport {
        let _p = tracing::info_span!("resolve_import", import_path = %import.path.display(self.db.upcast(), Edition::LATEST))
            .entered();
        tracing::debug!("resolving import: {:?} ({:?})", import, self.def_map.data.edition);
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

        if res.from_differing_crate {
            return PartialResolvedImport::Resolved(
                def.filter_visibility(|v| matches!(v, Visibility::Public)),
            );
        }

        // Check whether all namespaces are resolved.
        if def.is_full() {
            PartialResolvedImport::Resolved(def)
        } else {
            PartialResolvedImport::Indeterminate(def)
        }
    }

    fn record_resolved_import(&mut self, directive: &ImportDirective) {
        let _p = tracing::info_span!("record_resolved_import").entered();

        let module_id = directive.module_id;
        let import = &directive.import;
        let mut def = directive.status.namespaces();
        let vis = self
            .def_map
            .resolve_visibility(self.db, module_id, &directive.import.visibility, false)
            .unwrap_or(Visibility::Public);

        match import.source {
            ImportSource {
                kind: kind @ (ImportKind::Plain | ImportKind::TypeOnly),
                id,
                use_tree,
                ..
            } => {
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

                if kind == ImportKind::TypeOnly {
                    def.values = None;
                    def.macros = None;
                }
                let imp = ImportType::Import(ImportId { import: id, idx: use_tree });
                tracing::debug!("resolved import {:?} ({:?}) to {:?}", name, import, def);

                self.update(module_id, &[(name.cloned(), def)], vis, Some(imp));
            }
            ImportSource { kind: ImportKind::Glob, id, is_prelude, .. } => {
                tracing::debug!("glob import: {:?}", import);
                match def.take_types() {
                    Some(ModuleDefId::ModuleId(m)) => {
                        if is_prelude {
                            // Note: This dodgily overrides the injected prelude. The rustc
                            // implementation seems to work the same though.
                            cov_mark::hit!(std_prelude);
                            self.def_map.prelude = Some((m, Some(id)));
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

                            self.update(module_id, &items, vis, Some(ImportType::Glob(id)));
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

                            self.update(module_id, &items, vis, Some(ImportType::Glob(id)));
                            // record the glob import in case we add further items
                            let glob = self.glob_imports.entry(m.local_id).or_default();
                            if !glob.iter().any(|(mid, _, _)| *mid == module_id) {
                                glob.push((module_id, vis, id));
                            }
                        }
                    }
                    Some(ModuleDefId::AdtId(AdtId::EnumId(e))) => {
                        cov_mark::hit!(glob_enum);
                        // glob import from enum => just import all the variants

                        // We need to check if the def map the enum is from is us, if it is we can't
                        // call the def-map query since we are currently constructing it!
                        let loc = e.lookup(self.db);
                        let tree = loc.id.item_tree(self.db);
                        let current_def_map = self.def_map.krate == loc.container.krate
                            && self.def_map.block_id() == loc.container.block;
                        let def_map;
                        let resolutions = if current_def_map {
                            &self.def_map.enum_definitions[&e]
                        } else {
                            def_map = loc.container.def_map(self.db);
                            &def_map.enum_definitions[&e]
                        }
                        .iter()
                        .map(|&variant| {
                            let name = tree[variant.lookup(self.db).id.value].name.clone();
                            let res = PerNs::both(variant.into(), variant.into(), vis, None);
                            (Some(name), res)
                        })
                        .collect::<Vec<_>>();
                        self.update(module_id, &resolutions, vis, Some(ImportType::Glob(id)));
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
        import: Option<ImportType>,
    ) {
        self.db.unwind_if_cancelled();
        self.update_recursive(module_id, resolutions, vis, import, 0)
    }

    fn update_recursive(
        &mut self,
        // The module for which `resolutions` have been resolved.
        module_id: LocalModuleId,
        resolutions: &[(Option<Name>, PerNs)],
        // All resolutions are imported with this visibility; the visibilities in
        // the `PerNs` values are ignored and overwritten
        vis: Visibility,
        import: Option<ImportType>,
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
                    changed |=
                        self.push_res_and_update_glob_vis(module_id, name, *res, vis, import);
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
            .filter(|(glob_importing_module, _, _)| {
                // we know all resolutions have the same visibility (`vis`), so we
                // just need to check that once
                vis.is_visible_from_def_map(self.db, &self.def_map, *glob_importing_module)
            })
            .cloned()
            .collect::<Vec<_>>();

        for (glob_importing_module, glob_import_vis, use_) in glob_imports {
            let vis = glob_import_vis.min(vis, &self.def_map).unwrap_or(glob_import_vis);
            self.update_recursive(
                glob_importing_module,
                resolutions,
                vis,
                Some(ImportType::Glob(use_)),
                depth + 1,
            );
        }
    }

    fn push_res_and_update_glob_vis(
        &mut self,
        module_id: LocalModuleId,
        name: &Name,
        mut defs: PerNs,
        vis: Visibility,
        def_import_type: Option<ImportType>,
    ) -> bool {
        // `extern crate crate_name` things can be re-exported as `pub use crate_name`.
        // But they cannot be re-exported as `pub use self::crate_name`, `pub use crate::crate_name`
        // or `pub use ::crate_name`.
        //
        // This has been historically allowed, but may be not allowed in future
        // https://github.com/rust-lang/rust/issues/127909
        if let Some((_, v, it)) = defs.types.as_mut() {
            let is_extern_crate_reimport_without_prefix = || {
                let Some(ImportOrExternCrate::ExternCrate(_)) = it else {
                    return false;
                };
                let Some(ImportType::Import(id)) = def_import_type else {
                    return false;
                };
                let use_id = id.import.lookup(self.db).id;
                let item_tree = use_id.item_tree(self.db);
                let use_kind = item_tree[use_id.value].use_tree.kind();
                let UseTreeKind::Single { path, .. } = use_kind else {
                    return false;
                };
                path.segments().len() < 2
            };
            if is_extern_crate_reimport_without_prefix() {
                *v = vis;
            } else {
                *v = v.min(vis, &self.def_map).unwrap_or(vis);
            }
        }
        if let Some((_, v, _)) = defs.values.as_mut() {
            *v = v.min(vis, &self.def_map).unwrap_or(vis);
        }
        if let Some((_, v, _)) = defs.macros.as_mut() {
            *v = v.min(vis, &self.def_map).unwrap_or(vis);
        }

        let mut changed = false;

        if let Some(ImportType::Glob(_)) = def_import_type {
            let prev_defs = self.def_map[module_id].scope.get(name);

            // Multiple globs may import the same item and they may override visibility from
            // previously resolved globs. Handle overrides here and leave the rest to
            // `ItemScope::push_res_with_import()`.
            if let Some((def, def_vis, _)) = defs.types {
                if let Some((prev_def, prev_vis, _)) = prev_defs.types {
                    if def == prev_def
                        && self.from_glob_import.contains_type(module_id, name.clone())
                        && def_vis != prev_vis
                        && def_vis.max(prev_vis, &self.def_map) == Some(def_vis)
                    {
                        changed = true;
                        // This import is being handled here, don't pass it down to
                        // `ItemScope::push_res_with_import()`.
                        defs.types = None;
                        self.def_map.modules[module_id]
                            .scope
                            .update_visibility_types(name, def_vis);
                    }
                }
            }

            if let Some((def, def_vis, _)) = defs.values {
                if let Some((prev_def, prev_vis, _)) = prev_defs.values {
                    if def == prev_def
                        && self.from_glob_import.contains_value(module_id, name.clone())
                        && def_vis != prev_vis
                        && def_vis.max(prev_vis, &self.def_map) == Some(def_vis)
                    {
                        changed = true;
                        // See comment above.
                        defs.values = None;
                        self.def_map.modules[module_id]
                            .scope
                            .update_visibility_values(name, def_vis);
                    }
                }
            }

            if let Some((def, def_vis, _)) = defs.macros {
                if let Some((prev_def, prev_vis, _)) = prev_defs.macros {
                    if def == prev_def
                        && self.from_glob_import.contains_macro(module_id, name.clone())
                        && def_vis != prev_vis
                        && def_vis.max(prev_vis, &self.def_map) == Some(def_vis)
                    {
                        changed = true;
                        // See comment above.
                        defs.macros = None;
                        self.def_map.modules[module_id]
                            .scope
                            .update_visibility_macros(name, def_vis);
                    }
                }
            }
        }

        changed |= self.def_map.modules[module_id].scope.push_res_with_import(
            &mut self.from_glob_import,
            (module_id, name.clone()),
            defs,
            def_import_type,
        );

        changed
    }

    fn resolve_macros(&mut self) -> ReachedFixedPoint {
        let mut macros = mem::take(&mut self.unresolved_macros);
        let mut resolved = Vec::new();
        let mut push_resolved = |directive: &MacroDirective, call_id| {
            resolved.push((directive.module_id, directive.depth, directive.container, call_id));
        };

        #[derive(PartialEq, Eq)]
        enum Resolved {
            Yes,
            No,
        }

        let mut res = ReachedFixedPoint::Yes;
        // Retain unresolved macros after this round of resolution.
        let mut retain = |directive: &MacroDirective| {
            let subns = match &directive.kind {
                MacroDirectiveKind::FnLike { .. } => MacroSubNs::Bang,
                MacroDirectiveKind::Attr { .. } | MacroDirectiveKind::Derive { .. } => {
                    MacroSubNs::Attr
                }
            };
            let resolver = |path: &_| {
                let resolved_res = self.def_map.resolve_path_fp_with_macro(
                    self.db,
                    ResolveMode::Other,
                    directive.module_id,
                    path,
                    BuiltinShadowMode::Module,
                    Some(subns),
                );
                resolved_res.resolved_def.take_macros().map(|it| (it, self.db.macro_def(it)))
            };
            let resolver_def_id = |path: &_| resolver(path).map(|(_, it)| it);

            match &directive.kind {
                MacroDirectiveKind::FnLike { ast_id, expand_to, ctxt: call_site } => {
                    let call_id = macro_call_as_call_id(
                        self.db.upcast(),
                        ast_id,
                        *call_site,
                        *expand_to,
                        self.def_map.krate,
                        resolver_def_id,
                    );
                    if let Ok(Some(call_id)) = call_id {
                        self.def_map.modules[directive.module_id]
                            .scope
                            .add_macro_invoc(ast_id.ast_id, call_id);

                        push_resolved(directive, call_id);

                        res = ReachedFixedPoint::No;
                        return Resolved::Yes;
                    }
                }
                MacroDirectiveKind::Derive {
                    ast_id,
                    derive_attr,
                    derive_pos,
                    ctxt: call_site,
                    derive_macro_id,
                } => {
                    let id = derive_macro_as_call_id(
                        self.db,
                        ast_id,
                        *derive_attr,
                        *derive_pos as u32,
                        *call_site,
                        self.def_map.krate,
                        resolver,
                        *derive_macro_id,
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
                        return Resolved::Yes;
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
                        Resolved::Yes
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

                    let def = match resolver_def_id(path) {
                        Some(def) if def.is_attribute() => def,
                        _ => return Resolved::No,
                    };

                    // Skip #[test]/#[bench]/#[test_case] expansion, which would merely result in more memory usage
                    // due to duplicating functions into macro expansions, but only if `cfg(test)` is active,
                    // otherwise they are expanded to nothing and this can impact e.g. diagnostics (due to things
                    // being cfg'ed out).
                    // Ideally we will just expand them to nothing here. But we are only collecting macro calls,
                    // not expanding them, so we have no way to do that.
                    if matches!(
                        def.kind,
                        MacroDefKind::BuiltInAttr(_, expander)
                        if expander.is_test() || expander.is_bench() || expander.is_test_case()
                    ) {
                        let test_is_active =
                            self.cfg_options.check_atom(&CfgAtom::Flag(sym::test.clone()));
                        if test_is_active {
                            return recollect_without(self);
                        }
                    }

                    let call_id = || {
                        attr_macro_as_call_id(self.db, file_ast_id, attr, self.def_map.krate, def)
                    };
                    if matches!(def,
                        MacroDefId { kind: MacroDefKind::BuiltInAttr(_, exp), .. }
                        if exp.is_derive()
                    ) {
                        // Resolved to `#[derive]`, we don't actually expand this attribute like
                        // normal (as that would just be an identity expansion with extra output)
                        // Instead we treat derive attributes special and apply them separately.

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

                        match attr.parse_path_comma_token_tree(self.db.upcast()) {
                            Some(derive_macros) => {
                                let call_id = call_id();
                                let mut len = 0;
                                for (idx, (path, call_site)) in derive_macros.enumerate() {
                                    let ast_id = AstIdWithPath::new(
                                        file_id,
                                        ast_id.value,
                                        Interned::new(path),
                                    );
                                    self.unresolved_macros.push(MacroDirective {
                                        module_id: directive.module_id,
                                        depth: directive.depth + 1,
                                        kind: MacroDirectiveKind::Derive {
                                            ast_id,
                                            derive_attr: attr.id,
                                            derive_pos: idx,
                                            ctxt: call_site.ctx,
                                            derive_macro_id: call_id,
                                        },
                                        container: directive.container,
                                    });
                                    len = idx;
                                }

                                // We treat the #[derive] macro as an attribute call, but we do not resolve it for nameres collection.
                                // This is just a trick to be able to resolve the input to derives
                                // as proper paths in `Semantics`.
                                // Check the comment in [`builtin_attr_macro`].
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

                    if let MacroDefKind::ProcMacro(_, exp, _) = def.kind {
                        // If there's no expander for the proc macro (e.g.
                        // because proc macros are disabled, or building the
                        // proc macro crate failed), report this and skip
                        // expansion like we would if it was disabled
                        if let Some(err) = exp.as_expand_error(def.krate) {
                            self.def_map.diagnostics.push(DefDiagnostic::macro_error(
                                directive.module_id,
                                ast_id,
                                (**path).clone(),
                                err,
                            ));
                            return recollect_without(self);
                        }
                    }

                    let call_id = call_id();
                    self.def_map.modules[directive.module_id]
                        .scope
                        .add_attr_macro_invoc(ast_id, call_id);

                    push_resolved(directive, call_id);
                    res = ReachedFixedPoint::No;
                    return Resolved::Yes;
                }
            }

            Resolved::No
        };
        macros.retain(|it| retain(it) == Resolved::No);
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
        let recursion_limit = self.def_map.recursion_limit() as usize;
        let recursion_limit = Limit::new(if cfg!(test) {
            // Without this, `body::tests::your_stack_belongs_to_me` stack-overflows in debug
            std::cmp::min(32, recursion_limit)
        } else {
            recursion_limit
        });
        if recursion_limit.check(depth).is_err() {
            cov_mark::hit!(macro_expansion_overflow);
            tracing::warn!("macro expansion is too deep");
            return;
        }
        let file_id = macro_call_id.as_file();

        let item_tree = self.db.file_item_tree(file_id);

        let mod_dir = if macro_call_id.as_macro_file().is_include_macro(self.db.upcast()) {
            ModDir::root()
        } else {
            self.mod_dirs[&module_id].clone()
        };

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

        let _p = tracing::info_span!("DefCollector::finish").entered();

        for directive in &self.unresolved_macros {
            match &directive.kind {
                MacroDirectiveKind::FnLike { ast_id, expand_to, ctxt: call_site } => {
                    // FIXME: we shouldn't need to re-resolve the macro here just to get the unresolved error!
                    let macro_call_as_call_id = macro_call_as_call_id(
                        self.db.upcast(),
                        ast_id,
                        *call_site,
                        *expand_to,
                        self.def_map.krate,
                        |path| {
                            let resolved_res = self.def_map.resolve_path_fp_with_macro(
                                self.db,
                                ResolveMode::Other,
                                directive.module_id,
                                path,
                                BuiltinShadowMode::Module,
                                Some(MacroSubNs::Bang),
                            );
                            resolved_res.resolved_def.take_macros().map(|it| self.db.macro_def(it))
                        },
                    );
                    if let Err(UnresolvedMacro { path }) = macro_call_as_call_id {
                        self.def_map.diagnostics.push(DefDiagnostic::unresolved_macro_call(
                            directive.module_id,
                            MacroCallKind::FnLike {
                                ast_id: ast_id.ast_id,
                                expand_to: *expand_to,
                                eager: None,
                            },
                            path,
                        ));
                    }
                }
                MacroDirectiveKind::Derive {
                    ast_id,
                    derive_attr,
                    derive_pos,
                    derive_macro_id,
                    ..
                } => {
                    self.def_map.diagnostics.push(DefDiagnostic::unresolved_macro_call(
                        directive.module_id,
                        MacroCallKind::Derive {
                            ast_id: ast_id.ast_id,
                            derive_attr_index: *derive_attr,
                            derive_index: *derive_pos as u32,
                            derive_macro_id: *derive_macro_id,
                        },
                        ast_id.path.as_ref().clone(),
                    ));
                }
                // These are diagnosed by `reseed_with_unresolved_attribute`, as that function consumes them
                MacroDirectiveKind::Attr { .. } => {}
            }
        }

        // Emit diagnostics for all remaining unresolved imports.
        for import in &self.unresolved_imports {
            let &ImportDirective {
                module_id,
                import:
                    Import {
                        ref path,
                        source: ImportSource { use_tree, id, is_prelude: _, kind: _ },
                        ..
                    },
                ..
            } = import;
            if matches!(
                (path.segments().first(), &path.kind),
                (Some(krate), PathKind::Plain | PathKind::Abs) if self.unresolved_extern_crates.contains(krate)
            ) {
                continue;
            }
            let item_tree_id = id.lookup(self.db).id;
            self.def_map.diagnostics.push(DefDiagnostic::unresolved_import(
                module_id,
                item_tree_id,
                use_tree,
            ));
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
        let is_crate_root =
            self.module_id == DefMap::ROOT && self.def_collector.def_map.block.is_none();

        // Note: don't assert that inserted value is fresh: it's simply not true
        // for macros.
        self.def_collector.mod_dirs.insert(self.module_id, self.mod_dir.clone());

        // Prelude module is always considered to be `#[macro_use]`.
        if let Some((prelude_module, _use)) = self.def_collector.def_map.prelude {
            // Don't insert macros from the prelude into blocks, as they can be shadowed by other macros.
            if prelude_module.krate != krate && is_crate_root {
                cov_mark::hit!(prelude_is_macro_use);
                self.def_collector.import_macros_from_extern_crate(
                    prelude_module.krate,
                    None,
                    None,
                );
            }
        }
        let db = self.def_collector.db;
        let module_id = self.module_id;
        let update_def =
            |def_collector: &mut DefCollector<'_>, id, name: &Name, vis, has_constructor| {
                def_collector.def_map.modules[module_id].scope.declare(id);
                def_collector.update(
                    module_id,
                    &[(Some(name.clone()), PerNs::from_def(id, vis, has_constructor, None))],
                    vis,
                    None,
                )
            };
        let resolve_vis = |def_map: &DefMap, visibility| {
            def_map
                .resolve_visibility(db, module_id, visibility, false)
                .unwrap_or(Visibility::Public)
        };

        let mut process_mod_item = |item: ModItem| {
            let attrs = self.item_tree.attrs(db, krate, item.into());
            if let Some(cfg) = attrs.cfg() {
                if !self.is_cfg_enabled(&cfg) {
                    self.emit_unconfigured_diagnostic(self.tree_id, item.into(), &cfg);
                    return;
                }
            }

            if let Err(()) = self.resolve_attributes(&attrs, item, container) {
                // Do not process the item. It has at least one non-builtin attribute, so the
                // fixed-point algorithm is required to resolve the rest of them.
                return;
            }

            let module = self.def_collector.def_map.module_id(module_id);
            let def_map = &mut self.def_collector.def_map;

            match item {
                ModItem::Mod(m) => self.collect_module(m, &attrs),
                ModItem::Use(item_tree_id) => {
                    let id = UseLoc {
                        container: module,
                        id: ItemTreeId::new(self.tree_id, item_tree_id),
                    }
                    .intern(db);
                    let is_prelude = attrs.by_key(&sym::prelude_import).exists();
                    Import::from_use(
                        self.item_tree,
                        ItemTreeId::new(self.tree_id, item_tree_id),
                        id,
                        is_prelude,
                        |import| {
                            self.def_collector.unresolved_imports.push(ImportDirective {
                                module_id: self.module_id,
                                import,
                                status: PartialResolvedImport::Unresolved,
                            });
                        },
                    )
                }
                ModItem::ExternCrate(item_tree_id) => {
                    let id = ExternCrateLoc {
                        container: module,
                        id: ItemTreeId::new(self.tree_id, item_tree_id),
                    }
                    .intern(db);
                    def_map.modules[self.module_id].scope.define_extern_crate_decl(id);

                    let item_tree::ExternCrate { name, visibility, alias, ast_id } =
                        &self.item_tree[item_tree_id];

                    let is_self = *name == sym::self_;
                    let resolved = if is_self {
                        cov_mark::hit!(extern_crate_self_as);
                        Some(def_map.crate_root())
                    } else {
                        self.def_collector
                            .deps
                            .get(name)
                            .map(|dep| CrateRootModuleId { krate: dep.crate_id })
                    };

                    let name = match alias {
                        Some(ImportAlias::Alias(name)) => Some(name),
                        Some(ImportAlias::Underscore) => None,
                        None => Some(name),
                    };

                    if let Some(resolved) = resolved {
                        let vis = resolve_vis(def_map, &self.item_tree[*visibility]);

                        if is_crate_root {
                            // extern crates in the crate root are special-cased to insert entries into the extern prelude: rust-lang/rust#54658
                            if let Some(name) = name {
                                Arc::get_mut(&mut def_map.data)
                                    .unwrap()
                                    .extern_prelude
                                    .insert(name.clone(), (resolved, Some(id)));
                            }
                            // they also allow `#[macro_use]`
                            if !is_self {
                                self.process_macro_use_extern_crate(
                                    id,
                                    attrs.by_key(&sym::macro_use).attrs(),
                                    resolved.krate,
                                );
                            }
                        }

                        self.def_collector.update(
                            module_id,
                            &[(
                                name.cloned(),
                                PerNs::types(
                                    resolved.into(),
                                    vis,
                                    Some(ImportOrExternCrate::ExternCrate(id)),
                                ),
                            )],
                            vis,
                            Some(ImportType::ExternCrate(id)),
                        );
                    } else {
                        if let Some(name) = name {
                            self.def_collector.unresolved_extern_crates.insert(name.clone());
                        }
                        self.def_collector.def_map.diagnostics.push(
                            DefDiagnostic::unresolved_extern_crate(
                                module_id,
                                InFile::new(self.file_id(), *ast_id),
                            ),
                        );
                    }
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
                ModItem::Macro2(id) => self.collect_macro_def(id, module),
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

                    if self.def_collector.def_map.block.is_none()
                        && self.def_collector.is_proc_macro
                        && self.module_id == DefMap::ROOT
                    {
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
                        !matches!(it.shape, FieldsShape::Record),
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
                    let enum_ =
                        EnumLoc { container: module, id: ItemTreeId::new(self.tree_id, id) }
                            .intern(db);

                    let vis = resolve_vis(def_map, &self.item_tree[it.visibility]);
                    update_def(self.def_collector, enum_.into(), &it.name, vis, false);

                    let mut index = 0;
                    let variants = FileItemTreeId::range_iter(it.variants.clone())
                        .filter_map(|variant| {
                            let is_enabled = self
                                .item_tree
                                .attrs(db, krate, variant.into())
                                .cfg()
                                .and_then(|cfg| self.is_cfg_enabled(&cfg).not().then_some(cfg))
                                .map_or(Ok(()), Err);
                            match is_enabled {
                                Err(cfg) => {
                                    self.emit_unconfigured_diagnostic(
                                        self.tree_id,
                                        variant.into(),
                                        &cfg,
                                    );
                                    None
                                }
                                Ok(()) => Some({
                                    let loc = EnumVariantLoc {
                                        id: ItemTreeId::new(self.tree_id, variant),
                                        parent: enum_,
                                        index,
                                    }
                                    .intern(db);
                                    index += 1;
                                    loc
                                }),
                            }
                        })
                        .collect();
                    self.def_collector.def_map.enum_definitions.insert(enum_, variants);
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
        };

        // extern crates should be processed eagerly instead of deferred to resolving.
        // `#[macro_use] extern crate` is hoisted to imports macros before collecting
        // any other items.
        if is_crate_root {
            items
                .iter()
                .filter(|it| matches!(it, ModItem::ExternCrate(..)))
                .copied()
                .for_each(&mut process_mod_item);
            items
                .iter()
                .filter(|it| !matches!(it, ModItem::ExternCrate(..)))
                .copied()
                .for_each(process_mod_item);
        } else {
            items.iter().copied().for_each(process_mod_item);
        }
    }

    fn process_macro_use_extern_crate<'a>(
        &mut self,
        extern_crate_id: ExternCrateId,
        macro_use_attrs: impl Iterator<Item = &'a Attr>,
        target_crate: CrateId,
    ) {
        cov_mark::hit!(macro_rules_from_other_crates_are_visible_with_macro_use);
        let mut single_imports = Vec::new();
        for attr in macro_use_attrs {
            let Some(paths) = attr.parse_path_comma_token_tree(self.def_collector.db.upcast())
            else {
                // `#[macro_use]` (without any paths) found, forget collected names and just import
                // all visible macros.
                self.def_collector.import_macros_from_extern_crate(
                    target_crate,
                    None,
                    Some(extern_crate_id),
                );
                return;
            };
            for (path, _) in paths {
                if let Some(name) = path.as_ident() {
                    single_imports.push(name.clone());
                }
            }
        }

        self.def_collector.import_macros_from_extern_crate(
            target_crate,
            Some(single_imports),
            Some(extern_crate_id),
        );
    }

    fn collect_module(&mut self, module_id: FileItemTreeId<Mod>, attrs: &Attrs) {
        let path_attr = attrs.by_key(&sym::path).string_value_unescape();
        let is_macro_use = attrs.by_key(&sym::macro_use).exists();
        let module = &self.item_tree[module_id];
        match &module.kind {
            // inline module, just recurse
            ModKind::Inline { items } => {
                let module_id = self.push_child_module(
                    module.name.clone(),
                    module.ast_id,
                    None,
                    &self.item_tree[module.visibility],
                    module_id,
                );

                let Some(mod_dir) =
                    self.mod_dir.descend_into_definition(&module.name, path_attr.as_deref())
                else {
                    return;
                };
                ModCollector {
                    def_collector: &mut *self.def_collector,
                    macro_depth: self.macro_depth,
                    module_id,
                    tree_id: self.tree_id,
                    item_tree: self.item_tree,
                    mod_dir,
                }
                .collect_in_top_module(items);
                if is_macro_use {
                    self.import_all_legacy_macros(module_id);
                }
            }
            // out of line module, resolve, parse and recurse
            ModKind::Outline => {
                let ast_id = AstId::new(self.file_id(), module.ast_id);
                let db = self.def_collector.db;
                match self.mod_dir.resolve_declaration(
                    db,
                    self.file_id(),
                    &module.name,
                    path_attr.as_deref(),
                ) {
                    Ok((file_id, is_mod_rs, mod_dir)) => {
                        let item_tree = db.file_item_tree(file_id.into());
                        let krate = self.def_collector.def_map.krate;
                        let is_enabled = item_tree
                            .top_level_attrs(db, krate)
                            .cfg()
                            .and_then(|cfg| self.is_cfg_enabled(&cfg).not().then_some(cfg))
                            .map_or(Ok(()), Err);
                        match is_enabled {
                            Err(cfg) => {
                                self.emit_unconfigured_diagnostic(
                                    self.tree_id,
                                    AttrOwner::ModItem(module_id.into()),
                                    &cfg,
                                );
                            }
                            Ok(()) => {
                                let module_id = self.push_child_module(
                                    module.name.clone(),
                                    ast_id.value,
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
                                        .by_key(&sym::macro_use)
                                        .exists();
                                if is_macro_use {
                                    self.import_all_legacy_macros(module_id);
                                }
                            }
                        }
                    }
                    Err(candidates) => {
                        self.push_child_module(
                            module.name.clone(),
                            ast_id.value,
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
        declaration: FileAstId<ast::Module>,
        definition: Option<(EditionedFileId, bool)>,
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
            &[(Some(name), PerNs::from_def(def, vis, false, None))],
            vis,
            None,
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
                attr.path.display(self.def_collector.db.upcast(), Edition::LATEST)
            );

            let ast_id = AstIdWithPath::new(
                self.file_id(),
                mod_item.ast_id(self.item_tree),
                attr.path.clone(),
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

        let export_attr = attrs.by_key(&sym::macro_export);

        let is_export = export_attr.exists();
        let local_inner = if is_export {
            export_attr.tt_values().flat_map(|it| it.token_trees.iter()).any(|it| match it {
                tt::TokenTree::Leaf(tt::Leaf::Ident(ident)) => ident.sym == sym::local_inner_macros,
                _ => false,
            })
        } else {
            false
        };

        // Case 1: builtin macros
        let expander = if attrs.by_key(&sym::rustc_builtin_macro).exists() {
            // `#[rustc_builtin_macro = "builtin_name"]` overrides the `macro_rules!` name.
            let name;
            let name = match attrs.by_key(&sym::rustc_builtin_macro).string_value_with_span() {
                Some((it, span)) => {
                    name = Name::new_symbol(it.clone(), span.ctx);
                    &name
                }
                None => {
                    let explicit_name =
                        attrs.by_key(&sym::rustc_builtin_macro).tt_values().next().and_then(|tt| {
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
        let allow_internal_unsafe = attrs.by_key(&sym::allow_internal_unsafe).exists();

        let mut flags = MacroRulesLocFlags::empty();
        flags.set(MacroRulesLocFlags::LOCAL_INNER, local_inner);
        flags.set(MacroRulesLocFlags::ALLOW_INTERNAL_UNSAFE, allow_internal_unsafe);

        let macro_id = MacroRulesLoc {
            container: module,
            id: ItemTreeId::new(self.tree_id, id),
            flags,
            expander,
            edition: self.def_collector.def_map.data.edition,
        }
        .intern(self.def_collector.db);
        self.def_collector.define_macro_rules(
            self.module_id,
            mac.name.clone(),
            macro_id,
            is_export,
        );
    }

    fn collect_macro_def(&mut self, id: FileItemTreeId<Macro2>, module: ModuleId) {
        let krate = self.def_collector.def_map.krate;
        let mac = &self.item_tree[id];
        let ast_id = InFile::new(self.file_id(), mac.ast_id.upcast());

        // Case 1: builtin macros
        let mut helpers_opt = None;
        let attrs = self.item_tree.attrs(self.def_collector.db, krate, ModItem::from(id).into());
        let expander = if attrs.by_key(&sym::rustc_builtin_macro).exists() {
            if let Some(expander) = find_builtin_macro(&mac.name) {
                match expander {
                    Either::Left(it) => MacroExpander::BuiltIn(it),
                    Either::Right(it) => MacroExpander::BuiltInEager(it),
                }
            } else if let Some(expander) = find_builtin_derive(&mac.name) {
                if let Some(attr) = attrs.by_key(&sym::rustc_builtin_macro).tt_values().next() {
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
                            mac.name.display(self.def_collector.db.upcast(), Edition::LATEST),
                            name.display(self.def_collector.db.upcast(), Edition::LATEST),
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
        let allow_internal_unsafe = attrs.by_key(&sym::allow_internal_unsafe).exists();

        let macro_id = Macro2Loc {
            container: module,
            id: ItemTreeId::new(self.tree_id, id),
            expander,
            allow_internal_unsafe,
            edition: self.def_collector.def_map.data.edition,
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
                    .insert(self.def_collector.db.macro_def(macro_id.into()), helpers);
            }
        }
    }

    fn collect_macro_call(
        &mut self,
        &MacroCall { ref path, ast_id, expand_to, ctxt }: &MacroCall,
        container: ItemContainerId,
    ) {
        let ast_id = AstIdWithPath::new(self.file_id(), ast_id, path.clone());
        let db = self.def_collector.db;

        // FIXME: Immediately expanding in "Case 1" is insufficient since "Case 2" may also define
        // new legacy macros that create textual scopes. We need a way to resolve names in textual
        // scopes without eager expansion.

        // Case 1: try to resolve macro calls with single-segment name and expand macro_rules
        if let Ok(res) = macro_call_as_call_id_with_eager(
            db.upcast(),
            ast_id.ast_id,
            &ast_id.path,
            ctxt,
            expand_to,
            self.def_collector.def_map.krate,
            |path| {
                path.as_ident().and_then(|name| {
                    let def_map = &self.def_collector.def_map;
                    def_map
                        .with_ancestor_maps(db, self.module_id, &mut |map, module| {
                            map[module].scope.get_legacy_macro(name)?.last().copied()
                        })
                        .or_else(|| def_map[self.module_id].scope.get(name).take_macros())
                        .or_else(|| Some(def_map.macro_use_prelude.get(name).copied()?.0))
                        .filter(|&id| {
                            sub_namespace_match(
                                Some(MacroSubNs::from_id(db, id)),
                                Some(MacroSubNs::Bang),
                            )
                        })
                        .map(|it| self.def_collector.db.macro_def(it))
                })
            },
            |path| {
                let resolved_res = self.def_collector.def_map.resolve_path_fp_with_macro(
                    db,
                    ResolveMode::Other,
                    self.module_id,
                    path,
                    BuiltinShadowMode::Module,
                    Some(MacroSubNs::Bang),
                );
                resolved_res.resolved_def.take_macros().map(|it| db.macro_def(it))
            },
        ) {
            // FIXME: if there were errors, this might've been in the eager expansion from an
            // unresolved macro, so we need to push this into late macro resolution. see fixme above
            if res.err.is_none() {
                // Legacy macros need to be expanded immediately, so that any macros they produce
                // are in scope.
                if let Some(call_id) = res.value {
                    self.def_collector.def_map.modules[self.module_id]
                        .scope
                        .add_macro_invoc(ast_id.ast_id, call_id);
                    self.def_collector.collect_macro_expansion(
                        self.module_id,
                        call_id,
                        self.macro_depth + 1,
                        container,
                    );
                }

                return;
            }
        }

        // Case 2: resolve in module scope, expand during name resolution.
        self.def_collector.unresolved_macros.push(MacroDirective {
            module_id: self.module_id,
            depth: self.macro_depth + 1,
            kind: MacroDirectiveKind::FnLike { ast_id, expand_to, ctxt },
            container,
        });
    }

    fn import_all_legacy_macros(&mut self, module_id: LocalModuleId) {
        let Some((source, target)) = Self::borrow_modules(
            self.def_collector.def_map.modules.as_mut(),
            module_id,
            self.module_id,
        ) else {
            return;
        };

        for (name, macs) in source.scope.legacy_macros() {
            if let Some(&mac) = macs.last() {
                target.scope.define_legacy_macro(name.clone(), mac);
            }
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

    fn emit_unconfigured_diagnostic(&mut self, tree_id: TreeId, item: AttrOwner, cfg: &CfgExpr) {
        self.def_collector.def_map.diagnostics.push(DefDiagnostic::unconfigured_code(
            self.module_id,
            tree_id,
            item,
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
    use base_db::SourceDatabase;
    use test_fixture::WithFixture;

    use crate::{nameres::DefMapCrateData, test_db::TestDB};

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
            proc_macros: Default::default(),
            from_glob_import: Default::default(),
            skip_attrs: Default::default(),
            is_proc_macro: false,
            unresolved_extern_crates: Default::default(),
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
        let def_map = DefMap::empty(
            krate,
            Arc::new(DefMapCrateData::new(edition)),
            ModuleData::new(module_origin, Visibility::Public),
            None,
        );
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
