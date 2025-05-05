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
//! phase. Macro calls are represented as a triple of `(Path, Option<Name>,
//! TokenTree)`.
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

pub mod assoc;
pub mod attr_resolution;
mod collector;
pub mod diagnostics;
mod mod_resolution;
mod path_resolution;
pub mod proc_macro;

#[cfg(test)]
mod tests;

use std::ops::Deref;

use base_db::Crate;
use hir_expand::{
    EditionedFileId, ErasedAstId, HirFileId, InFile, MacroCallId, MacroDefId, mod_path::ModPath,
    name::Name, proc_macro::ProcMacroKind,
};
use intern::Symbol;
use itertools::Itertools;
use la_arena::Arena;
use rustc_hash::{FxHashMap, FxHashSet};
use span::{Edition, FileAstId, FileId, ROOT_ERASED_FILE_AST_ID};
use stdx::format_to;
use syntax::{AstNode, SmolStr, SyntaxNode, ToSmolStr, ast};
use triomphe::Arc;
use tt::TextRange;

use crate::{
    AstId, BlockId, BlockLoc, CrateRootModuleId, ExternCrateId, FunctionId, FxIndexMap,
    LocalModuleId, Lookup, MacroExpander, MacroId, ModuleId, ProcMacroId, UseId,
    db::DefDatabase,
    item_scope::{BuiltinShadowMode, ItemScope},
    item_tree::{ItemTreeId, Mod, TreeId},
    nameres::{diagnostics::DefDiagnostic, path_resolution::ResolveMode},
    per_ns::PerNs,
    visibility::{Visibility, VisibilityExplicitness},
};

pub use self::path_resolution::ResolvePathResultPrefixInfo;

const PREDEFINED_TOOLS: &[SmolStr] = &[
    SmolStr::new_static("clippy"),
    SmolStr::new_static("rustfmt"),
    SmolStr::new_static("diagnostic"),
    SmolStr::new_static("miri"),
    SmolStr::new_static("rust_analyzer"),
];

/// Parts of the def map that are only needed when analyzing code in the same crate.
///
/// There are some data in the def map (e.g. extern prelude) that is only needed when analyzing
/// things in the same crate (and maybe in the IDE layer), e.g. the extern prelude. If we put
/// it in the DefMap dependant DefMaps will be invalidated when they change (e.g. when we add
/// a dependency to the crate). Instead we split them out of the DefMap into a LocalDefMap struct.
/// `crate_local_def_map()` returns both, and `crate_def_map()` returns only the external-relevant
/// DefMap.
#[derive(Debug, PartialEq, Eq, Default)]
pub struct LocalDefMap {
    // FIXME: There are probably some other things that could be here, but this is less severe and you
    // need to be careful with things that block def maps also have.
    /// The extern prelude which contains all root modules of external crates that are in scope.
    extern_prelude: FxIndexMap<Name, (CrateRootModuleId, Option<ExternCrateId>)>,
}

impl std::hash::Hash for LocalDefMap {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let LocalDefMap { extern_prelude } = self;
        extern_prelude.len().hash(state);
        for (name, (crate_root, extern_crate)) in extern_prelude {
            name.hash(state);
            crate_root.hash(state);
            extern_crate.hash(state);
        }
    }
}

impl LocalDefMap {
    pub(crate) const EMPTY: &Self =
        &Self { extern_prelude: FxIndexMap::with_hasher(rustc_hash::FxBuildHasher) };

    fn shrink_to_fit(&mut self) {
        let Self { extern_prelude } = self;
        extern_prelude.shrink_to_fit();
    }

    pub(crate) fn extern_prelude(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&Name, (CrateRootModuleId, Option<ExternCrateId>))> + '_
    {
        self.extern_prelude.iter().map(|(name, &def)| (name, def))
    }
}

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
    /// The crate this `DefMap` belongs to.
    krate: Crate,
    /// When this is a block def map, this will hold the block id of the block and module that
    /// contains this block.
    block: Option<BlockInfo>,
    /// The modules and their data declared in this crate.
    pub modules: Arena<ModuleData>,
    /// The prelude module for this crate. This either comes from an import
    /// marked with the `prelude_import` attribute, or (in the normal case) from
    /// a dependency (`std` or `core`).
    /// The prelude is empty for non-block DefMaps (unless `#[prelude_import]` was used,
    /// but that attribute is nightly and when used in a block, it affects resolution globally
    /// so we aren't handling this correctly anyways).
    prelude: Option<(ModuleId, Option<UseId>)>,
    /// `macro_use` prelude that contains macros from `#[macro_use]`'d external crates. Note that
    /// this contains all kinds of macro, not just `macro_rules!` macro.
    /// ExternCrateId being None implies it being imported from the general prelude import.
    macro_use_prelude: FxHashMap<Name, (MacroId, Option<ExternCrateId>)>,

    // FIXME: AstId's are fairly unstable
    /// Tracks which custom derives are in scope for an item, to allow resolution of derive helper
    /// attributes.
    // FIXME: Figure out a better way for the IDE layer to resolve these?
    derive_helpers_in_scope: FxHashMap<AstId<ast::Item>, Vec<(Name, MacroId, MacroCallId)>>,
    // FIXME: AstId's are fairly unstable
    /// A mapping from [`hir_expand::MacroDefId`] to [`crate::MacroId`].
    pub macro_def_to_macro_id: FxHashMap<ErasedAstId, MacroId>,

    /// The diagnostics that need to be emitted for this crate.
    diagnostics: Vec<DefDiagnostic>,

    /// The crate data that is shared between a crate's def map and all its block def maps.
    data: Arc<DefMapCrateData>,
}

/// Data that belongs to a crate which is shared between a crate's def map and all its block def maps.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DefMapCrateData {
    /// Side table for resolving derive helpers.
    exported_derives: FxHashMap<MacroDefId, Box<[Name]>>,
    fn_proc_macro_mapping: FxHashMap<FunctionId, ProcMacroId>,

    /// Custom attributes registered with `#![register_attr]`.
    registered_attrs: Vec<Symbol>,
    /// Custom tool modules registered with `#![register_tool]`.
    registered_tools: Vec<Symbol>,
    /// Unstable features of Rust enabled with `#![feature(A, B)]`.
    unstable_features: FxHashSet<Symbol>,
    /// #[rustc_coherence_is_core]
    rustc_coherence_is_core: bool,
    no_core: bool,
    no_std: bool,

    edition: Edition,
    recursion_limit: Option<u32>,
}

impl DefMapCrateData {
    fn new(edition: Edition) -> Self {
        Self {
            exported_derives: FxHashMap::default(),
            fn_proc_macro_mapping: FxHashMap::default(),
            registered_attrs: Vec::new(),
            registered_tools: PREDEFINED_TOOLS.iter().map(|it| Symbol::intern(it)).collect(),
            unstable_features: FxHashSet::default(),
            rustc_coherence_is_core: false,
            no_core: false,
            no_std: false,
            edition,
            recursion_limit: None,
        }
    }

    fn shrink_to_fit(&mut self) {
        let Self {
            exported_derives,
            fn_proc_macro_mapping,
            registered_attrs,
            registered_tools,
            unstable_features,
            rustc_coherence_is_core: _,
            no_core: _,
            no_std: _,
            edition: _,
            recursion_limit: _,
        } = self;
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
    fn def_map(self, db: &dyn DefDatabase, krate: Crate) -> &DefMap {
        self.into_module(krate).def_map(db)
    }

    fn into_module(self, krate: Crate) -> ModuleId {
        ModuleId { krate, block: self.block, local_id: self.local_id }
    }

    fn is_block_module(self) -> bool {
        self.block.is_some() && self.local_id == DefMap::ROOT
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
        definition: EditionedFileId,
    },
    /// Note that non-inline modules, by definition, live inside non-macro file.
    File {
        is_mod_rs: bool,
        declaration: FileAstId<ast::Module>,
        declaration_tree_id: ItemTreeId<Mod>,
        definition: EditionedFileId,
    },
    Inline {
        definition_tree_id: ItemTreeId<Mod>,
        definition: FileAstId<ast::Module>,
    },
    /// Pseudo-module introduced by a block scope (contains only inner items).
    BlockExpr {
        id: BlockId,
        block: AstId<ast::BlockExpr>,
    },
}

impl ModuleOrigin {
    pub fn declaration(&self) -> Option<AstId<ast::Module>> {
        match self {
            &ModuleOrigin::File { declaration, declaration_tree_id, .. } => {
                Some(AstId::new(declaration_tree_id.file_id(), declaration))
            }
            &ModuleOrigin::Inline { definition, definition_tree_id } => {
                Some(AstId::new(definition_tree_id.file_id(), definition))
            }
            ModuleOrigin::CrateRoot { .. } | ModuleOrigin::BlockExpr { .. } => None,
        }
    }

    pub fn file_id(&self) -> Option<EditionedFileId> {
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
    pub fn definition_source(&self, db: &dyn DefDatabase) -> InFile<ModuleSource> {
        match self {
            &ModuleOrigin::File { definition: editioned_file_id, .. }
            | &ModuleOrigin::CrateRoot { definition: editioned_file_id } => {
                let sf = db.parse(editioned_file_id).tree();
                InFile::new(editioned_file_id.into(), ModuleSource::SourceFile(sf))
            }
            &ModuleOrigin::Inline { definition, definition_tree_id } => InFile::new(
                definition_tree_id.file_id(),
                ModuleSource::Module(
                    AstId::new(definition_tree_id.file_id(), definition).to_node(db),
                ),
            ),
            ModuleOrigin::BlockExpr { block, .. } => {
                InFile::new(block.file_id, ModuleSource::BlockExpr(block.to_node(db)))
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
    /// Parent module in the same `DefMap`.
    ///
    /// [`None`] for block modules because they are always its `DefMap`'s root.
    pub parent: Option<LocalModuleId>,
    pub children: FxIndexMap<Name, LocalModuleId>,
    pub scope: ItemScope,
}

#[inline]
pub fn crate_def_map(db: &dyn DefDatabase, crate_id: Crate) -> &DefMap {
    crate_local_def_map(db, crate_id).def_map(db)
}

#[allow(unused_lifetimes)]
mod __ {
    use super::*;
    #[salsa_macros::tracked]
    pub(crate) struct DefMapPair<'db> {
        #[tracked]
        #[return_ref]
        pub(crate) def_map: DefMap,
        #[return_ref]
        pub(crate) local: LocalDefMap,
    }
}
pub(crate) use __::DefMapPair;

#[salsa_macros::tracked(return_ref)]
pub(crate) fn crate_local_def_map(db: &dyn DefDatabase, crate_id: Crate) -> DefMapPair<'_> {
    let krate = crate_id.data(db);
    let _p = tracing::info_span!(
        "crate_def_map_query",
        name=?crate_id
            .extra_data(db)
            .display_name
            .as_ref()
            .map(|it| it.crate_name().to_smolstr())
            .unwrap_or_default()
    )
    .entered();

    let module_data = ModuleData::new(
        ModuleOrigin::CrateRoot { definition: krate.root_file_id(db) },
        Visibility::Public,
    );

    let def_map =
        DefMap::empty(crate_id, Arc::new(DefMapCrateData::new(krate.edition)), module_data, None);
    let (def_map, local_def_map) = collector::collect_defs(
        db,
        def_map,
        TreeId::new(krate.root_file_id(db).into(), None),
        None,
    );

    DefMapPair::new(db, def_map, local_def_map)
}

#[salsa_macros::tracked(return_ref)]
pub fn block_def_map(db: &dyn DefDatabase, block_id: BlockId) -> DefMap {
    let BlockLoc { ast_id, module } = block_id.lookup(db);

    let visibility = Visibility::Module(
        ModuleId { krate: module.krate, local_id: DefMap::ROOT, block: module.block },
        VisibilityExplicitness::Implicit,
    );
    let module_data =
        ModuleData::new(ModuleOrigin::BlockExpr { block: ast_id, id: block_id }, visibility);

    let local_def_map = crate_local_def_map(db, module.krate);
    let def_map = DefMap::empty(
        module.krate,
        local_def_map.def_map(db).data.clone(),
        module_data,
        Some(BlockInfo {
            block: block_id,
            parent: BlockRelativeModuleId { block: module.block, local_id: module.local_id },
        }),
    );

    let (def_map, _) = collector::collect_defs(
        db,
        def_map,
        TreeId::new(ast_id.file_id, Some(block_id)),
        Some(local_def_map.local(db)),
    );
    def_map
}

impl DefMap {
    /// The module id of a crate or block root.
    pub const ROOT: LocalModuleId = LocalModuleId::from_raw(la_arena::RawIdx::from_u32(0));

    pub fn edition(&self) -> Edition {
        self.data.edition
    }

    fn empty(
        krate: Crate,
        crate_data: Arc<DefMapCrateData>,
        module_data: ModuleData,
        block: Option<BlockInfo>,
    ) -> DefMap {
        let mut modules: Arena<ModuleData> = Arena::default();
        let root = modules.alloc(module_data);
        assert_eq!(root, Self::ROOT);

        DefMap {
            block,
            modules,
            krate,
            prelude: None,
            macro_use_prelude: FxHashMap::default(),
            derive_helpers_in_scope: FxHashMap::default(),
            diagnostics: Vec::new(),
            data: crate_data,
            macro_def_to_macro_id: FxHashMap::default(),
        }
    }
    fn shrink_to_fit(&mut self) {
        // Exhaustive match to require handling new fields.
        let Self {
            macro_use_prelude,
            diagnostics,
            modules,
            derive_helpers_in_scope,
            block: _,
            krate: _,
            prelude: _,
            data: _,
            macro_def_to_macro_id,
        } = self;

        macro_def_to_macro_id.shrink_to_fit();
        macro_use_prelude.shrink_to_fit();
        diagnostics.shrink_to_fit();
        modules.shrink_to_fit();
        derive_helpers_in_scope.shrink_to_fit();
        for (_, module) in modules.iter_mut() {
            module.children.shrink_to_fit();
            module.scope.shrink_to_fit();
        }
    }
}

impl DefMap {
    pub fn modules_for_file<'a>(
        &'a self,
        db: &'a dyn DefDatabase,
        file_id: FileId,
    ) -> impl Iterator<Item = LocalModuleId> + 'a {
        self.modules
            .iter()
            .filter(move |(_id, data)| {
                data.origin.file_id().map(|file_id| file_id.file_id(db)) == Some(file_id)
            })
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

    pub fn registered_tools(&self) -> &[Symbol] {
        &self.data.registered_tools
    }

    pub fn registered_attrs(&self) -> &[Symbol] {
        &self.data.registered_attrs
    }

    pub fn is_unstable_feature_enabled(&self, feature: &Symbol) -> bool {
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

    pub fn krate(&self) -> Crate {
        self.krate
    }

    pub fn module_id(&self, local_id: LocalModuleId) -> ModuleId {
        let block = self.block.map(|b| b.block);
        ModuleId { krate: self.krate, local_id, block }
    }

    pub fn crate_root(&self) -> CrateRootModuleId {
        CrateRootModuleId { krate: self.krate }
    }

    /// This is the same as [`Self::crate_root`] for crate def maps, but for block def maps, it
    /// returns the root block module.
    pub fn root_module_id(&self) -> ModuleId {
        self.module_id(Self::ROOT)
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

    /// Get a reference to the def map's diagnostics.
    pub fn diagnostics(&self) -> &[DefDiagnostic] {
        self.diagnostics.as_slice()
    }

    pub fn recursion_limit(&self) -> u32 {
        // 128 is the default in rustc
        self.data.recursion_limit.unwrap_or(128)
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
            current_map = arc;
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

            map.modules[module].scope.dump(db, buf);

            for (name, child) in
                map.modules[module].children.iter().sorted_by(|a, b| Ord::cmp(&a.0, &b.0))
            {
                let path = format!("{path}::{}", name.display(db, Edition::LATEST));
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
            current_map = arc;
        }

        format_to!(buf, "crate scope\n");
        buf
    }
}

impl DefMap {
    pub(crate) fn block_id(&self) -> Option<BlockId> {
        self.block.map(|block| block.block)
    }

    pub(crate) fn prelude(&self) -> Option<(ModuleId, Option<UseId>)> {
        self.prelude
    }

    pub(crate) fn macro_use_prelude(&self) -> &FxHashMap<Name, (MacroId, Option<ExternCrateId>)> {
        &self.macro_use_prelude
    }

    pub(crate) fn resolve_path(
        &self,
        local_def_map: &LocalDefMap,
        db: &dyn DefDatabase,
        original_module: LocalModuleId,
        path: &ModPath,
        shadow: BuiltinShadowMode,
        expected_macro_subns: Option<MacroSubNs>,
    ) -> (PerNs, Option<usize>) {
        let res = self.resolve_path_fp_with_macro(
            local_def_map,
            db,
            ResolveMode::Other,
            original_module,
            path,
            shadow,
            expected_macro_subns,
        );
        (res.resolved_def, res.segment_index)
    }

    /// The first `Option<usize>` points at the `Enum` segment in case of `Enum::Variant`, the second
    /// points at the unresolved segments.
    pub(crate) fn resolve_path_locally(
        &self,
        local_def_map: &LocalDefMap,
        db: &dyn DefDatabase,
        original_module: LocalModuleId,
        path: &ModPath,
        shadow: BuiltinShadowMode,
    ) -> (PerNs, Option<usize>, ResolvePathResultPrefixInfo) {
        let res = self.resolve_path_fp_with_macro_single(
            local_def_map,
            db,
            ResolveMode::Other,
            original_module,
            path,
            shadow,
            None, // Currently this function isn't used for macro resolution.
        );
        (res.resolved_def, res.segment_index, res.prefix_info)
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
            if let Some(it) = f(parent, block_info.parent.local_id) {
                return Some(it);
            }
            block = parent.block;
        }

        None
    }
}

impl ModuleData {
    pub(crate) fn new(origin: ModuleOrigin, visibility: Visibility) -> Self {
        ModuleData {
            origin,
            visibility,
            parent: None,
            children: Default::default(),
            scope: ItemScope::default(),
        }
    }

    /// Returns a node which defines this module. That is, a file or a `mod foo {}` with items.
    pub fn definition_source(&self, db: &dyn DefDatabase) -> InFile<ModuleSource> {
        self.origin.definition_source(db)
    }

    /// Same as [`definition_source`] but only returns the file id to prevent parsing the ASt.
    pub fn definition_source_file_id(&self) -> HirFileId {
        match self.origin {
            ModuleOrigin::File { definition, .. } | ModuleOrigin::CrateRoot { definition } => {
                definition.into()
            }
            ModuleOrigin::Inline { definition_tree_id, .. } => definition_tree_id.file_id(),
            ModuleOrigin::BlockExpr { block, .. } => block.file_id,
        }
    }

    pub fn definition_source_range(&self, db: &dyn DefDatabase) -> InFile<TextRange> {
        match &self.origin {
            &ModuleOrigin::File { definition, .. } | &ModuleOrigin::CrateRoot { definition } => {
                InFile::new(
                    definition.into(),
                    ErasedAstId::new(definition.into(), ROOT_ERASED_FILE_AST_ID).to_range(db),
                )
            }
            &ModuleOrigin::Inline { definition, definition_tree_id } => InFile::new(
                definition_tree_id.file_id(),
                AstId::new(definition_tree_id.file_id(), definition).to_range(db),
            ),
            ModuleOrigin::BlockExpr { block, .. } => InFile::new(block.file_id, block.to_range(db)),
        }
    }

    /// Returns a node which declares this module, either a `mod foo;` or a `mod foo {}`.
    /// `None` for the crate root or block.
    pub fn declaration_source(&self, db: &dyn DefDatabase) -> Option<InFile<ast::Module>> {
        let decl = self.origin.declaration()?;
        let value = decl.to_node(db);
        Some(InFile { file_id: decl.file_id, value })
    }

    /// Returns the range which declares this module, either a `mod foo;` or a `mod foo {}`.
    /// `None` for the crate root or block.
    pub fn declaration_source_range(&self, db: &dyn DefDatabase) -> Option<InFile<TextRange>> {
        let decl = self.origin.declaration()?;
        Some(InFile { file_id: decl.file_id, value: decl.to_range(db) })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleSource {
    SourceFile(ast::SourceFile),
    Module(ast::Module),
    BlockExpr(ast::BlockExpr),
}

impl ModuleSource {
    pub fn node(&self) -> SyntaxNode {
        match self {
            ModuleSource::SourceFile(it) => it.syntax().clone(),
            ModuleSource::Module(it) => it.syntax().clone(),
            ModuleSource::BlockExpr(it) => it.syntax().clone(),
        }
    }
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
                    ProcMacroKind::Bang => Self::Bang,
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
