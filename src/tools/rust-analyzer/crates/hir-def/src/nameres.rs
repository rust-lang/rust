//! This module implements import-resolution/macro expansion algorithm.
//!
//! The result of this module is `DefMap`: a data structure which contains:
//!
//!   * a tree of modules for the crate
//!   * for each module, a set of items visible in the module (directly declared
//!     or imported)
//!
//! Note that `DefMap` contains fully macro expanded code.
//!
//! Computing `DefMap` can be partitioned into several logically
//! independent "phases". The phases are mutually recursive though, there's no
//! strict ordering.
//!
//! ## Collecting RawItems
//!
//! This happens in the `raw` module, which parses a single source file into a
//! set of top-level items. Nested imports are desugared to flat imports in this
//! phase. Macro calls are represented as a triple of (Path, Option<Name>,
//! TokenTree).
//!
//! ## Collecting Modules
//!
//! This happens in the `collector` module. In this phase, we recursively walk
//! tree of modules, collect raw items from submodules, populate module scopes
//! with defined items (so, we assign item ids in this phase) and record the set
//! of unresolved imports and macros.
//!
//! While we walk tree of modules, we also record macro_rules definitions and
//! expand calls to macro_rules defined macros.
//!
//! ## Resolving Imports
//!
//! We maintain a list of currently unresolved imports. On every iteration, we
//! try to resolve some imports from this list. If the import is resolved, we
//! record it, by adding an item to current module scope and, if necessary, by
//! recursively populating glob imports.
//!
//! ## Resolving Macros
//!
//! macro_rules from the same crate use a global mutable namespace. We expand
//! them immediately, when we collect modules.
//!
//! Macros from other crates (including proc-macros) can be used with
//! `foo::bar!` syntax. We handle them similarly to imports. There's a list of
//! unexpanded macros. On every iteration, we try to resolve each macro call
//! path and, upon success, we run macro expansion and "collect module" phase on
//! the result

pub mod attr_resolution;
pub mod proc_macro;
pub mod diagnostics;
mod collector;
mod mod_resolution;
mod path_resolution;

#[cfg(test)]
mod tests;

use std::{cmp::Ord, ops::Deref};

use base_db::{CrateId, Edition, FileId, ProcMacroKind};
use hir_expand::{name::Name, InFile, MacroCallId, MacroDefId};
use itertools::Itertools;
use la_arena::Arena;
use profile::Count;
use rustc_hash::{FxHashMap, FxHashSet};
use stdx::format_to;
use syntax::{ast, SmolStr};
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    item_scope::{BuiltinShadowMode, ItemScope},
    item_tree::{ItemTreeId, Mod, TreeId},
    nameres::{diagnostics::DefDiagnostic, path_resolution::ResolveMode},
    path::ModPath,
    per_ns::PerNs,
    visibility::Visibility,
    AstId, BlockId, BlockLoc, CrateRootModuleId, FunctionId, LocalModuleId, Lookup, MacroExpander,
    MacroId, ModuleId, ProcMacroId,
};

/// Contains the results of (early) name resolution.
///
/// A `DefMap` stores the module tree and the definitions that are in scope in every module after
/// item-level macros have been expanded.
///
/// Every crate has a primary `DefMap` whose root is the crate's main file (`main.rs`/`lib.rs`),
/// computed by the `crate_def_map` query. Additionally, every block expression introduces the
/// opportunity to write arbitrary item and module hierarchies, and thus gets its own `DefMap` that
/// is computed by the `block_def_map` query.
#[derive(Debug, PartialEq, Eq)]
pub struct DefMap {
    _c: Count<Self>,
    /// When this is a block def map, this will hold the block id of the the block and module that
    /// contains this block.
    block: Option<BlockInfo>,
    /// The modules and their data declared in this crate.
    modules: Arena<ModuleData>,
    krate: CrateId,
    /// The prelude module for this crate. This either comes from an import
    /// marked with the `prelude_import` attribute, or (in the normal case) from
    /// a dependency (`std` or `core`).
    /// The prelude is empty for non-block DefMaps (unless `#[prelude_import]` was used,
    /// but that attribute is nightly and when used in a block, it affects resolution globally
    /// so we aren't handling this correctly anyways).
    prelude: Option<ModuleId>,
    /// `macro_use` prelude that contains macros from `#[macro_use]`'d external crates. Note that
    /// this contains all kinds of macro, not just `macro_rules!` macro.
    macro_use_prelude: FxHashMap<Name, MacroId>,

    /// Tracks which custom derives are in scope for an item, to allow resolution of derive helper
    /// attributes.
    derive_helpers_in_scope: FxHashMap<AstId<ast::Item>, Vec<(Name, MacroId, MacroCallId)>>,

    /// The diagnostics that need to be emitted for this crate.
    diagnostics: Vec<DefDiagnostic>,

    /// The crate data that is shared between a crate's def map and all its block def maps.
    data: Arc<DefMapCrateData>,
}

/// Data that belongs to a crate which is shared between a crate's def map and all its block def maps.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DefMapCrateData {
    /// The extern prelude which contains all root modules of external crates that are in scope.
    extern_prelude: FxHashMap<Name, CrateRootModuleId>,

    /// Side table for resolving derive helpers.
    exported_derives: FxHashMap<MacroDefId, Box<[Name]>>,
    fn_proc_macro_mapping: FxHashMap<FunctionId, ProcMacroId>,
    /// The error that occurred when failing to load the proc-macro dll.
    proc_macro_loading_error: Option<Box<str>>,

    /// Custom attributes registered with `#![register_attr]`.
    registered_attrs: Vec<SmolStr>,
    /// Custom tool modules registered with `#![register_tool]`.
    registered_tools: Vec<SmolStr>,
    /// Unstable features of Rust enabled with `#![feature(A, B)]`.
    unstable_features: FxHashSet<SmolStr>,
    /// #[rustc_coherence_is_core]
    rustc_coherence_is_core: bool,
    no_core: bool,
    no_std: bool,

    edition: Edition,
    recursion_limit: Option<u32>,
}

impl DefMapCrateData {
    fn shrink_to_fit(&mut self) {
        let Self {
            extern_prelude,
            exported_derives,
            fn_proc_macro_mapping,
            registered_attrs,
            registered_tools,
            unstable_features,
            proc_macro_loading_error: _,
            rustc_coherence_is_core: _,
            no_core: _,
            no_std: _,
            edition: _,
            recursion_limit: _,
        } = self;
        extern_prelude.shrink_to_fit();
        exported_derives.shrink_to_fit();
        fn_proc_macro_mapping.shrink_to_fit();
        registered_attrs.shrink_to_fit();
        registered_tools.shrink_to_fit();
        unstable_features.shrink_to_fit();
    }
}

/// For `DefMap`s computed for a block expression, this stores its location in the parent map.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct BlockInfo {
    /// The `BlockId` this `DefMap` was created from.
    block: BlockId,
    /// The containing module.
    parent: BlockRelativeModuleId,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct BlockRelativeModuleId {
    block: Option<BlockId>,
    local_id: LocalModuleId,
}

impl BlockRelativeModuleId {
    fn def_map(self, db: &dyn DefDatabase, krate: CrateId) -> Arc<DefMap> {
        self.into_module(krate).def_map(db)
    }

    fn into_module(self, krate: CrateId) -> ModuleId {
        ModuleId { krate, block: self.block, local_id: self.local_id }
    }
}

impl std::ops::Index<LocalModuleId> for DefMap {
    type Output = ModuleData;
    fn index(&self, id: LocalModuleId) -> &ModuleData {
        &self.modules[id]
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum ModuleOrigin {
    CrateRoot {
        definition: FileId,
    },
    /// Note that non-inline modules, by definition, live inside non-macro file.
    File {
        is_mod_rs: bool,
        declaration: AstId<ast::Module>,
        declaration_tree_id: ItemTreeId<Mod>,
        definition: FileId,
    },
    Inline {
        definition_tree_id: ItemTreeId<Mod>,
        definition: AstId<ast::Module>,
    },
    /// Pseudo-module introduced by a block scope (contains only inner items).
    BlockExpr {
        block: AstId<ast::BlockExpr>,
    },
}

impl ModuleOrigin {
    pub fn declaration(&self) -> Option<AstId<ast::Module>> {
        match self {
            ModuleOrigin::File { declaration: module, .. }
            | ModuleOrigin::Inline { definition: module, .. } => Some(*module),
            ModuleOrigin::CrateRoot { .. } | ModuleOrigin::BlockExpr { .. } => None,
        }
    }

    pub fn file_id(&self) -> Option<FileId> {
        match self {
            ModuleOrigin::File { definition, .. } | ModuleOrigin::CrateRoot { definition } => {
                Some(*definition)
            }
            _ => None,
        }
    }

    pub fn is_inline(&self) -> bool {
        match self {
            ModuleOrigin::Inline { .. } | ModuleOrigin::BlockExpr { .. } => true,
            ModuleOrigin::CrateRoot { .. } | ModuleOrigin::File { .. } => false,
        }
    }

    /// Returns a node which defines this module.
    /// That is, a file or a `mod foo {}` with items.
    fn definition_source(&self, db: &dyn DefDatabase) -> InFile<ModuleSource> {
        match self {
            ModuleOrigin::File { definition, .. } | ModuleOrigin::CrateRoot { definition } => {
                let file_id = *definition;
                let sf = db.parse(file_id).tree();
                InFile::new(file_id.into(), ModuleSource::SourceFile(sf))
            }
            ModuleOrigin::Inline { definition, .. } => InFile::new(
                definition.file_id,
                ModuleSource::Module(definition.to_node(db.upcast())),
            ),
            ModuleOrigin::BlockExpr { block } => {
                InFile::new(block.file_id, ModuleSource::BlockExpr(block.to_node(db.upcast())))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ModuleData {
    /// Where does this module come from?
    pub origin: ModuleOrigin,
    /// Declared visibility of this module.
    pub visibility: Visibility,
    /// Always [`None`] for block modules
    pub parent: Option<LocalModuleId>,
    pub children: FxHashMap<Name, LocalModuleId>,
    pub scope: ItemScope,
}

impl DefMap {
    /// The module id of a crate or block root.
    pub const ROOT: LocalModuleId = LocalModuleId::from_raw(la_arena::RawIdx::from_u32(0));

    pub(crate) fn crate_def_map_query(db: &dyn DefDatabase, krate: CrateId) -> Arc<DefMap> {
        let _p = profile::span("crate_def_map_query").detail(|| {
            db.crate_graph()[krate].display_name.as_deref().unwrap_or_default().to_string()
        });

        let crate_graph = db.crate_graph();

        let edition = crate_graph[krate].edition;
        let origin = ModuleOrigin::CrateRoot { definition: crate_graph[krate].root_file_id };
        let def_map = DefMap::empty(krate, edition, ModuleData::new(origin, Visibility::Public));
        let def_map = collector::collect_defs(
            db,
            def_map,
            TreeId::new(crate_graph[krate].root_file_id.into(), None),
        );

        Arc::new(def_map)
    }

    pub(crate) fn block_def_map_query(db: &dyn DefDatabase, block_id: BlockId) -> Arc<DefMap> {
        let block: BlockLoc = db.lookup_intern_block(block_id);

        let tree_id = TreeId::new(block.ast_id.file_id, Some(block_id));

        let parent_map = block.module.def_map(db);
        let krate = block.module.krate;
        let local_id = LocalModuleId::from_raw(la_arena::RawIdx::from(0));
        // NB: we use `None` as block here, which would be wrong for implicit
        // modules declared by blocks with items. At the moment, we don't use
        // this visibility for anything outside IDE, so that's probably OK.
        let visibility = Visibility::Module(ModuleId { krate, local_id, block: None });
        let module_data =
            ModuleData::new(ModuleOrigin::BlockExpr { block: block.ast_id }, visibility);

        let mut def_map = DefMap::empty(krate, parent_map.data.edition, module_data);
        def_map.data = parent_map.data.clone();
        def_map.block = Some(BlockInfo {
            block: block_id,
            parent: BlockRelativeModuleId {
                block: block.module.block,
                local_id: block.module.local_id,
            },
        });

        let def_map = collector::collect_defs(db, def_map, tree_id);
        Arc::new(def_map)
    }

    fn empty(krate: CrateId, edition: Edition, module_data: ModuleData) -> DefMap {
        let mut modules: Arena<ModuleData> = Arena::default();
        let root = modules.alloc(module_data);
        assert_eq!(root, Self::ROOT);

        DefMap {
            _c: Count::new(),
            block: None,
            modules,
            krate,
            prelude: None,
            macro_use_prelude: FxHashMap::default(),
            derive_helpers_in_scope: FxHashMap::default(),
            diagnostics: Vec::new(),
            data: Arc::new(DefMapCrateData {
                extern_prelude: FxHashMap::default(),
                exported_derives: FxHashMap::default(),
                fn_proc_macro_mapping: FxHashMap::default(),
                proc_macro_loading_error: None,
                registered_attrs: Vec::new(),
                registered_tools: Vec::new(),
                unstable_features: FxHashSet::default(),
                rustc_coherence_is_core: false,
                no_core: false,
                no_std: false,
                edition,
                recursion_limit: None,
            }),
        }
    }

    pub fn modules_for_file(&self, file_id: FileId) -> impl Iterator<Item = LocalModuleId> + '_ {
        self.modules
            .iter()
            .filter(move |(_id, data)| data.origin.file_id() == Some(file_id))
            .map(|(id, _data)| id)
    }

    pub fn modules(&self) -> impl Iterator<Item = (LocalModuleId, &ModuleData)> + '_ {
        self.modules.iter()
    }

    pub fn derive_helpers_in_scope(
        &self,
        id: AstId<ast::Adt>,
    ) -> Option<&[(Name, MacroId, MacroCallId)]> {
        self.derive_helpers_in_scope.get(&id.map(|it| it.upcast())).map(Deref::deref)
    }

    pub fn registered_tools(&self) -> &[SmolStr] {
        &self.data.registered_tools
    }

    pub fn registered_attrs(&self) -> &[SmolStr] {
        &self.data.registered_attrs
    }

    pub fn is_unstable_feature_enabled(&self, feature: &str) -> bool {
        self.data.unstable_features.contains(feature)
    }

    pub fn is_rustc_coherence_is_core(&self) -> bool {
        self.data.rustc_coherence_is_core
    }

    pub fn is_no_std(&self) -> bool {
        self.data.no_std || self.data.no_core
    }

    pub fn fn_as_proc_macro(&self, id: FunctionId) -> Option<ProcMacroId> {
        self.data.fn_proc_macro_mapping.get(&id).copied()
    }

    pub fn proc_macro_loading_error(&self) -> Option<&str> {
        self.data.proc_macro_loading_error.as_deref()
    }

    pub fn krate(&self) -> CrateId {
        self.krate
    }

    pub(crate) fn block_id(&self) -> Option<BlockId> {
        self.block.map(|block| block.block)
    }

    pub(crate) fn prelude(&self) -> Option<ModuleId> {
        self.prelude
    }

    pub(crate) fn extern_prelude(&self) -> impl Iterator<Item = (&Name, ModuleId)> + '_ {
        self.data.extern_prelude.iter().map(|(name, &def)| (name, def.into()))
    }

    pub(crate) fn macro_use_prelude(&self) -> impl Iterator<Item = (&Name, MacroId)> + '_ {
        self.macro_use_prelude.iter().map(|(name, &def)| (name, def))
    }

    pub fn module_id(&self, local_id: LocalModuleId) -> ModuleId {
        let block = self.block.map(|b| b.block);
        ModuleId { krate: self.krate, local_id, block }
    }

    pub fn crate_root(&self) -> CrateRootModuleId {
        CrateRootModuleId { krate: self.krate }
    }

    pub(crate) fn resolve_path(
        &self,
        db: &dyn DefDatabase,
        original_module: LocalModuleId,
        path: &ModPath,
        shadow: BuiltinShadowMode,
        expected_macro_subns: Option<MacroSubNs>,
    ) -> (PerNs, Option<usize>) {
        let res = self.resolve_path_fp_with_macro(
            db,
            ResolveMode::Other,
            original_module,
            path,
            shadow,
            expected_macro_subns,
        );
        (res.resolved_def, res.segment_index)
    }

    pub(crate) fn resolve_path_locally(
        &self,
        db: &dyn DefDatabase,
        original_module: LocalModuleId,
        path: &ModPath,
        shadow: BuiltinShadowMode,
    ) -> (PerNs, Option<usize>) {
        let res = self.resolve_path_fp_with_macro_single(
            db,
            ResolveMode::Other,
            original_module,
            path,
            shadow,
            None, // Currently this function isn't used for macro resolution.
        );
        (res.resolved_def, res.segment_index)
    }

    /// Ascends the `DefMap` hierarchy and calls `f` with every `DefMap` and containing module.
    ///
    /// If `f` returns `Some(val)`, iteration is stopped and `Some(val)` is returned. If `f` returns
    /// `None`, iteration continues.
    pub(crate) fn with_ancestor_maps<T>(
        &self,
        db: &dyn DefDatabase,
        local_mod: LocalModuleId,
        f: &mut dyn FnMut(&DefMap, LocalModuleId) -> Option<T>,
    ) -> Option<T> {
        if let Some(it) = f(self, local_mod) {
            return Some(it);
        }
        let mut block = self.block;
        while let Some(block_info) = block {
            let parent = block_info.parent.def_map(db, self.krate);
            if let Some(it) = f(&parent, block_info.parent.local_id) {
                return Some(it);
            }
            block = parent.block;
        }

        None
    }

    /// If this `DefMap` is for a block expression, returns the module containing the block (which
    /// might again be a block, or a module inside a block).
    pub fn parent(&self) -> Option<ModuleId> {
        let BlockRelativeModuleId { block, local_id } = self.block?.parent;
        Some(ModuleId { krate: self.krate, block, local_id })
    }

    /// Returns the module containing `local_mod`, either the parent `mod`, or the module (or block) containing
    /// the block, if `self` corresponds to a block expression.
    pub fn containing_module(&self, local_mod: LocalModuleId) -> Option<ModuleId> {
        match self[local_mod].parent {
            Some(parent) => Some(self.module_id(parent)),
            None => {
                self.block.map(
                    |BlockInfo { parent: BlockRelativeModuleId { block, local_id }, .. }| {
                        ModuleId { krate: self.krate, block, local_id }
                    },
                )
            }
        }
    }

    // FIXME: this can use some more human-readable format (ideally, an IR
    // even), as this should be a great debugging aid.
    pub fn dump(&self, db: &dyn DefDatabase) -> String {
        let mut buf = String::new();
        let mut arc;
        let mut current_map = self;
        while let Some(block) = current_map.block {
            go(&mut buf, db, current_map, "block scope", Self::ROOT);
            buf.push('\n');
            arc = block.parent.def_map(db, self.krate);
            current_map = &arc;
        }
        go(&mut buf, db, current_map, "crate", Self::ROOT);
        return buf;

        fn go(
            buf: &mut String,
            db: &dyn DefDatabase,
            map: &DefMap,
            path: &str,
            module: LocalModuleId,
        ) {
            format_to!(buf, "{}\n", path);

            map.modules[module].scope.dump(db.upcast(), buf);

            for (name, child) in
                map.modules[module].children.iter().sorted_by(|a, b| Ord::cmp(&a.0, &b.0))
            {
                let path = format!("{path}::{}", name.display(db.upcast()));
                buf.push('\n');
                go(buf, db, map, &path, *child);
            }
        }
    }

    pub fn dump_block_scopes(&self, db: &dyn DefDatabase) -> String {
        let mut buf = String::new();
        let mut arc;
        let mut current_map = self;
        while let Some(block) = current_map.block {
            format_to!(buf, "{:?} in {:?}\n", block.block, block.parent);
            arc = block.parent.def_map(db, self.krate);
            current_map = &arc;
        }

        format_to!(buf, "crate scope\n");
        buf
    }

    fn shrink_to_fit(&mut self) {
        // Exhaustive match to require handling new fields.
        let Self {
            _c: _,
            macro_use_prelude,
            diagnostics,
            modules,
            derive_helpers_in_scope,
            block: _,
            krate: _,
            prelude: _,
            data: _,
        } = self;

        macro_use_prelude.shrink_to_fit();
        diagnostics.shrink_to_fit();
        modules.shrink_to_fit();
        derive_helpers_in_scope.shrink_to_fit();
        for (_, module) in modules.iter_mut() {
            module.children.shrink_to_fit();
            module.scope.shrink_to_fit();
        }
    }

    /// Get a reference to the def map's diagnostics.
    pub fn diagnostics(&self) -> &[DefDiagnostic] {
        self.diagnostics.as_slice()
    }

    pub fn recursion_limit(&self) -> Option<u32> {
        self.data.recursion_limit
    }
}

impl ModuleData {
    pub(crate) fn new(origin: ModuleOrigin, visibility: Visibility) -> Self {
        ModuleData {
            origin,
            visibility,
            parent: None,
            children: FxHashMap::default(),
            scope: ItemScope::default(),
        }
    }

    /// Returns a node which defines this module. That is, a file or a `mod foo {}` with items.
    pub fn definition_source(&self, db: &dyn DefDatabase) -> InFile<ModuleSource> {
        self.origin.definition_source(db)
    }

    /// Returns a node which declares this module, either a `mod foo;` or a `mod foo {}`.
    /// `None` for the crate root or block.
    pub fn declaration_source(&self, db: &dyn DefDatabase) -> Option<InFile<ast::Module>> {
        let decl = self.origin.declaration()?;
        let value = decl.to_node(db.upcast());
        Some(InFile { file_id: decl.file_id, value })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleSource {
    SourceFile(ast::SourceFile),
    Module(ast::Module),
    BlockExpr(ast::BlockExpr),
}

/// See `sub_namespace_match()`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MacroSubNs {
    /// Function-like macros, suffixed with `!`.
    Bang,
    /// Macros inside attributes, i.e. attribute macros and derive macros.
    Attr,
}

impl MacroSubNs {
    fn from_id(db: &dyn DefDatabase, macro_id: MacroId) -> Self {
        let expander = match macro_id {
            MacroId::Macro2Id(it) => it.lookup(db).expander,
            MacroId::MacroRulesId(it) => it.lookup(db).expander,
            MacroId::ProcMacroId(it) => {
                return match it.lookup(db).kind {
                    ProcMacroKind::CustomDerive | ProcMacroKind::Attr => Self::Attr,
                    ProcMacroKind::FuncLike => Self::Bang,
                };
            }
        };

        // Eager macros aren't *guaranteed* to be bang macros, but they *are* all bang macros currently.
        match expander {
            MacroExpander::Declarative
            | MacroExpander::BuiltIn(_)
            | MacroExpander::BuiltInEager(_) => Self::Bang,
            MacroExpander::BuiltInAttr(_) | MacroExpander::BuiltInDerive(_) => Self::Attr,
        }
    }
}

/// Quoted from [rustc]:
/// Macro namespace is separated into two sub-namespaces, one for bang macros and
/// one for attribute-like macros (attributes, derives).
/// We ignore resolutions from one sub-namespace when searching names in scope for another.
///
/// [rustc]: https://github.com/rust-lang/rust/blob/1.69.0/compiler/rustc_resolve/src/macros.rs#L75
fn sub_namespace_match(candidate: Option<MacroSubNs>, expected: Option<MacroSubNs>) -> bool {
    match (candidate, expected) {
        (Some(candidate), Some(expected)) => candidate == expected,
        _ => true,
    }
}
