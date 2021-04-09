//! This module implements import-resolution/macro expansion algorithm.
//!
//! The result of this module is `CrateDefMap`: a data structure which contains:
//!
//!   * a tree of modules for the crate
//!   * for each module, a set of items visible in the module (directly declared
//!     or imported)
//!
//! Note that `CrateDefMap` contains fully macro expanded code.
//!
//! Computing `CrateDefMap` can be partitioned into several logically
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

mod collector;
mod mod_resolution;
mod path_resolution;

#[cfg(test)]
mod tests;
mod proc_macro;

use std::sync::Arc;

use base_db::{CrateId, Edition, FileId};
use hir_expand::{diagnostics::DiagnosticSink, name::Name, InFile, MacroDefId};
use la_arena::Arena;
use profile::Count;
use rustc_hash::FxHashMap;
use stdx::format_to;
use syntax::ast;

use crate::{
    db::DefDatabase,
    item_scope::{BuiltinShadowMode, ItemScope},
    nameres::{diagnostics::DefDiagnostic, path_resolution::ResolveMode},
    path::ModPath,
    per_ns::PerNs,
    AstId, BlockId, BlockLoc, LocalModuleId, ModuleDefId, ModuleId,
};

use self::proc_macro::ProcMacroDef;

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
    block: Option<BlockInfo>,
    root: LocalModuleId,
    modules: Arena<ModuleData>,
    krate: CrateId,
    /// The prelude module for this crate. This either comes from an import
    /// marked with the `prelude_import` attribute, or (in the normal case) from
    /// a dependency (`std` or `core`).
    prelude: Option<ModuleId>,
    extern_prelude: FxHashMap<Name, ModuleDefId>,

    /// Side table with additional proc. macro info, for use by name resolution in downstream
    /// crates.
    ///
    /// (the primary purpose is to resolve derive helpers)
    exported_proc_macros: FxHashMap<MacroDefId, ProcMacroDef>,

    edition: Edition,
    diagnostics: Vec<DefDiagnostic>,
}

/// For `DefMap`s computed for a block expression, this stores its location in the parent map.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct BlockInfo {
    /// The `BlockId` this `DefMap` was created from.
    block: BlockId,
    /// The containing module.
    parent: ModuleId,
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
        definition: FileId,
    },
    Inline {
        definition: AstId<ast::Module>,
    },
    /// Pseudo-module introduced by a block scope (contains only inner items).
    BlockExpr {
        block: AstId<ast::BlockExpr>,
    },
}

impl Default for ModuleOrigin {
    fn default() -> Self {
        ModuleOrigin::CrateRoot { definition: FileId(0) }
    }
}

impl ModuleOrigin {
    fn declaration(&self) -> Option<AstId<ast::Module>> {
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
            ModuleOrigin::Inline { definition } => InFile::new(
                definition.file_id,
                ModuleSource::Module(definition.to_node(db.upcast())),
            ),
            ModuleOrigin::BlockExpr { block } => {
                InFile::new(block.file_id, ModuleSource::BlockExpr(block.to_node(db.upcast())))
            }
        }
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
pub struct ModuleData {
    pub parent: Option<LocalModuleId>,
    pub children: FxHashMap<Name, LocalModuleId>,
    pub scope: ItemScope,

    /// Where does this module come from?
    pub origin: ModuleOrigin,
}

impl DefMap {
    pub(crate) fn crate_def_map_query(db: &dyn DefDatabase, krate: CrateId) -> Arc<DefMap> {
        let _p = profile::span("crate_def_map_query").detail(|| {
            db.crate_graph()[krate].display_name.as_deref().unwrap_or_default().to_string()
        });
        let edition = db.crate_graph()[krate].edition;
        let def_map = DefMap::empty(krate, edition);
        let def_map = collector::collect_defs(db, def_map, None);
        Arc::new(def_map)
    }

    pub(crate) fn block_def_map_query(
        db: &dyn DefDatabase,
        block_id: BlockId,
    ) -> Option<Arc<DefMap>> {
        let block: BlockLoc = db.lookup_intern_block(block_id);

        let item_tree = db.file_item_tree(block.ast_id.file_id);
        if item_tree.inner_items_of_block(block.ast_id.value).is_empty() {
            return None;
        }

        let block_info = BlockInfo { block: block_id, parent: block.module };

        let parent_map = block.module.def_map(db);
        let mut def_map = DefMap::empty(block.module.krate, parent_map.edition);
        def_map.block = Some(block_info);

        let def_map = collector::collect_defs(db, def_map, Some(block.ast_id));
        Some(Arc::new(def_map))
    }

    fn empty(krate: CrateId, edition: Edition) -> DefMap {
        let mut modules: Arena<ModuleData> = Arena::default();
        let root = modules.alloc(ModuleData::default());
        DefMap {
            _c: Count::new(),
            block: None,
            krate,
            edition,
            extern_prelude: FxHashMap::default(),
            exported_proc_macros: FxHashMap::default(),
            prelude: None,
            root,
            modules,
            diagnostics: Vec::new(),
        }
    }

    pub fn add_diagnostics(
        &self,
        db: &dyn DefDatabase,
        module: LocalModuleId,
        sink: &mut DiagnosticSink,
    ) {
        self.diagnostics.iter().for_each(|it| it.add_to(db, module, sink))
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

    pub fn root(&self) -> LocalModuleId {
        self.root
    }

    pub(crate) fn krate(&self) -> CrateId {
        self.krate
    }

    pub(crate) fn block_id(&self) -> Option<BlockId> {
        self.block.as_ref().map(|block| block.block)
    }

    pub(crate) fn prelude(&self) -> Option<ModuleId> {
        self.prelude
    }

    pub(crate) fn extern_prelude(&self) -> impl Iterator<Item = (&Name, &ModuleDefId)> + '_ {
        self.extern_prelude.iter()
    }

    pub fn module_id(&self, local_id: LocalModuleId) -> ModuleId {
        let block = self.block.as_ref().map(|b| b.block);
        ModuleId { krate: self.krate, local_id, block }
    }

    pub(crate) fn crate_root(&self, db: &dyn DefDatabase) -> ModuleId {
        self.with_ancestor_maps(db, self.root, &mut |def_map, _module| {
            if def_map.block.is_none() {
                Some(def_map.module_id(def_map.root))
            } else {
                None
            }
        })
        .expect("DefMap chain without root")
    }

    pub(crate) fn resolve_path(
        &self,
        db: &dyn DefDatabase,
        original_module: LocalModuleId,
        path: &ModPath,
        shadow: BuiltinShadowMode,
    ) -> (PerNs, Option<usize>) {
        let res =
            self.resolve_path_fp_with_macro(db, ResolveMode::Other, original_module, path, shadow);
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
        );
        (res.resolved_def, res.segment_index)
    }

    /// Ascends the `DefMap` hierarchy and calls `f` with every `DefMap` and containing module.
    ///
    /// If `f` returns `Some(val)`, iteration is stopped and `Some(val)` is returned. If `f` returns
    /// `None`, iteration continues.
    pub fn with_ancestor_maps<T>(
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
            let parent = block_info.parent.def_map(db);
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
        Some(self.block?.parent)
    }

    /// Returns the module containing `local_mod`, either the parent `mod`, or the module containing
    /// the block, if `self` corresponds to a block expression.
    pub fn containing_module(&self, local_mod: LocalModuleId) -> Option<ModuleId> {
        match &self[local_mod].parent {
            Some(parent) => Some(self.module_id(*parent)),
            None => match &self.block {
                Some(block) => Some(block.parent),
                None => None,
            },
        }
    }

    // FIXME: this can use some more human-readable format (ideally, an IR
    // even), as this should be a great debugging aid.
    pub fn dump(&self, db: &dyn DefDatabase) -> String {
        let mut buf = String::new();
        let mut arc;
        let mut current_map = self;
        while let Some(block) = &current_map.block {
            go(&mut buf, current_map, "block scope", current_map.root);
            buf.push('\n');
            arc = block.parent.def_map(db);
            current_map = &*arc;
        }
        go(&mut buf, current_map, "crate", current_map.root);
        return buf;

        fn go(buf: &mut String, map: &DefMap, path: &str, module: LocalModuleId) {
            format_to!(buf, "{}\n", path);

            map.modules[module].scope.dump(buf);

            for (name, child) in map.modules[module].children.iter() {
                let path = format!("{}::{}", path, name);
                buf.push('\n');
                go(buf, map, &path, *child);
            }
        }
    }

    fn shrink_to_fit(&mut self) {
        // Exhaustive match to require handling new fields.
        let Self {
            _c: _,
            exported_proc_macros,
            extern_prelude,
            diagnostics,
            modules,
            block: _,
            edition: _,
            krate: _,
            prelude: _,
            root: _,
        } = self;

        extern_prelude.shrink_to_fit();
        exported_proc_macros.shrink_to_fit();
        diagnostics.shrink_to_fit();
        modules.shrink_to_fit();
        for (_, module) in modules.iter_mut() {
            module.children.shrink_to_fit();
            module.scope.shrink_to_fit();
        }
    }
}

impl ModuleData {
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

mod diagnostics {
    use cfg::{CfgExpr, CfgOptions};
    use hir_expand::diagnostics::DiagnosticSink;
    use hir_expand::hygiene::Hygiene;
    use hir_expand::{InFile, MacroCallKind};
    use syntax::ast::AttrsOwner;
    use syntax::{ast, AstNode, AstPtr, SyntaxKind, SyntaxNodePtr};

    use crate::path::ModPath;
    use crate::{db::DefDatabase, diagnostics::*, nameres::LocalModuleId, AstId};

    #[derive(Debug, PartialEq, Eq)]
    enum DiagnosticKind {
        UnresolvedModule { declaration: AstId<ast::Module>, candidate: String },

        UnresolvedExternCrate { ast: AstId<ast::ExternCrate> },

        UnresolvedImport { ast: AstId<ast::Use>, index: usize },

        UnconfiguredCode { ast: AstId<ast::Item>, cfg: CfgExpr, opts: CfgOptions },

        UnresolvedProcMacro { ast: MacroCallKind },

        UnresolvedMacroCall { ast: AstId<ast::MacroCall> },

        MacroError { ast: MacroCallKind, message: String },
    }

    #[derive(Debug, PartialEq, Eq)]
    pub(super) struct DefDiagnostic {
        in_module: LocalModuleId,
        kind: DiagnosticKind,
    }

    impl DefDiagnostic {
        pub(super) fn unresolved_module(
            container: LocalModuleId,
            declaration: AstId<ast::Module>,
            candidate: String,
        ) -> Self {
            Self {
                in_module: container,
                kind: DiagnosticKind::UnresolvedModule { declaration, candidate },
            }
        }

        pub(super) fn unresolved_extern_crate(
            container: LocalModuleId,
            declaration: AstId<ast::ExternCrate>,
        ) -> Self {
            Self {
                in_module: container,
                kind: DiagnosticKind::UnresolvedExternCrate { ast: declaration },
            }
        }

        pub(super) fn unresolved_import(
            container: LocalModuleId,
            ast: AstId<ast::Use>,
            index: usize,
        ) -> Self {
            Self { in_module: container, kind: DiagnosticKind::UnresolvedImport { ast, index } }
        }

        pub(super) fn unconfigured_code(
            container: LocalModuleId,
            ast: AstId<ast::Item>,
            cfg: CfgExpr,
            opts: CfgOptions,
        ) -> Self {
            Self { in_module: container, kind: DiagnosticKind::UnconfiguredCode { ast, cfg, opts } }
        }

        pub(super) fn unresolved_proc_macro(container: LocalModuleId, ast: MacroCallKind) -> Self {
            Self { in_module: container, kind: DiagnosticKind::UnresolvedProcMacro { ast } }
        }

        pub(super) fn macro_error(
            container: LocalModuleId,
            ast: MacroCallKind,
            message: String,
        ) -> Self {
            Self { in_module: container, kind: DiagnosticKind::MacroError { ast, message } }
        }

        pub(super) fn unresolved_macro_call(
            container: LocalModuleId,
            ast: AstId<ast::MacroCall>,
        ) -> Self {
            Self { in_module: container, kind: DiagnosticKind::UnresolvedMacroCall { ast } }
        }

        pub(super) fn add_to(
            &self,
            db: &dyn DefDatabase,
            target_module: LocalModuleId,
            sink: &mut DiagnosticSink,
        ) {
            if self.in_module != target_module {
                return;
            }

            match &self.kind {
                DiagnosticKind::UnresolvedModule { declaration, candidate } => {
                    let decl = declaration.to_node(db.upcast());
                    sink.push(UnresolvedModule {
                        file: declaration.file_id,
                        decl: AstPtr::new(&decl),
                        candidate: candidate.clone(),
                    })
                }

                DiagnosticKind::UnresolvedExternCrate { ast } => {
                    let item = ast.to_node(db.upcast());
                    sink.push(UnresolvedExternCrate {
                        file: ast.file_id,
                        item: AstPtr::new(&item),
                    });
                }

                DiagnosticKind::UnresolvedImport { ast, index } => {
                    let use_item = ast.to_node(db.upcast());
                    let hygiene = Hygiene::new(db.upcast(), ast.file_id);
                    let mut cur = 0;
                    let mut tree = None;
                    ModPath::expand_use_item(
                        InFile::new(ast.file_id, use_item),
                        &hygiene,
                        |_mod_path, use_tree, _is_glob, _alias| {
                            if cur == *index {
                                tree = Some(use_tree.clone());
                            }

                            cur += 1;
                        },
                    );

                    if let Some(tree) = tree {
                        sink.push(UnresolvedImport { file: ast.file_id, node: AstPtr::new(&tree) });
                    }
                }

                DiagnosticKind::UnconfiguredCode { ast, cfg, opts } => {
                    let item = ast.to_node(db.upcast());
                    sink.push(InactiveCode {
                        file: ast.file_id,
                        node: AstPtr::new(&item).into(),
                        cfg: cfg.clone(),
                        opts: opts.clone(),
                    });
                }

                DiagnosticKind::UnresolvedProcMacro { ast } => {
                    let mut precise_location = None;
                    let (file, ast, name) = match ast {
                        MacroCallKind::FnLike { ast_id } => {
                            let node = ast_id.to_node(db.upcast());
                            (ast_id.file_id, SyntaxNodePtr::from(AstPtr::new(&node)), None)
                        }
                        MacroCallKind::Derive { ast_id, derive_name, .. } => {
                            let node = ast_id.to_node(db.upcast());

                            // Compute the precise location of the macro name's token in the derive
                            // list.
                            // FIXME: This does not handle paths to the macro, but neither does the
                            // rest of r-a.
                            let derive_attrs =
                                node.attrs().filter_map(|attr| match attr.as_simple_call() {
                                    Some((name, args)) if name == "derive" => Some(args),
                                    _ => None,
                                });
                            'outer: for attr in derive_attrs {
                                let tokens =
                                    attr.syntax().children_with_tokens().filter_map(|elem| {
                                        match elem {
                                            syntax::NodeOrToken::Node(_) => None,
                                            syntax::NodeOrToken::Token(tok) => Some(tok),
                                        }
                                    });
                                for token in tokens {
                                    if token.kind() == SyntaxKind::IDENT
                                        && token.text() == derive_name.as_str()
                                    {
                                        precise_location = Some(token.text_range());
                                        break 'outer;
                                    }
                                }
                            }

                            (
                                ast_id.file_id,
                                SyntaxNodePtr::from(AstPtr::new(&node)),
                                Some(derive_name.clone()),
                            )
                        }
                    };
                    sink.push(UnresolvedProcMacro {
                        file,
                        node: ast,
                        precise_location,
                        macro_name: name,
                    });
                }

                DiagnosticKind::UnresolvedMacroCall { ast } => {
                    let node = ast.to_node(db.upcast());
                    sink.push(UnresolvedMacroCall { file: ast.file_id, node: AstPtr::new(&node) });
                }

                DiagnosticKind::MacroError { ast, message } => {
                    let (file, ast) = match ast {
                        MacroCallKind::FnLike { ast_id, .. } => {
                            let node = ast_id.to_node(db.upcast());
                            (ast_id.file_id, SyntaxNodePtr::from(AstPtr::new(&node)))
                        }
                        MacroCallKind::Derive { ast_id, .. } => {
                            let node = ast_id.to_node(db.upcast());
                            (ast_id.file_id, SyntaxNodePtr::from(AstPtr::new(&node)))
                        }
                    };
                    sink.push(MacroError { file, node: ast, message: message.clone() });
                }
            }
        }
    }
}
