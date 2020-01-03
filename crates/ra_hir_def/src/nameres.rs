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

pub(crate) mod raw;
mod collector;
mod mod_resolution;
mod path_resolution;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use hir_expand::{diagnostics::DiagnosticSink, name::Name, InFile};
use ra_arena::Arena;
use ra_db::{CrateId, Edition, FileId, FilePosition};
use ra_prof::profile;
use ra_syntax::{
    ast::{self, AstNode},
    SyntaxNode,
};
use rustc_hash::FxHashMap;

use crate::{
    db::DefDatabase,
    item_scope::{BuiltinShadowMode, ItemScope},
    nameres::{diagnostics::DefDiagnostic, path_resolution::ResolveMode},
    path::ModPath,
    per_ns::PerNs,
    AstId, LocalModuleId, ModuleDefId, ModuleId,
};

/// Contains all top-level defs from a macro-expanded crate
#[derive(Debug, PartialEq, Eq)]
pub struct CrateDefMap {
    pub root: LocalModuleId,
    pub modules: Arena<LocalModuleId, ModuleData>,
    pub(crate) krate: CrateId,
    /// The prelude module for this crate. This either comes from an import
    /// marked with the `prelude_import` attribute, or (in the normal case) from
    /// a dependency (`std` or `core`).
    pub(crate) prelude: Option<ModuleId>,
    pub(crate) extern_prelude: FxHashMap<Name, ModuleDefId>,

    edition: Edition,
    diagnostics: Vec<DefDiagnostic>,
}

impl std::ops::Index<LocalModuleId> for CrateDefMap {
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
        declaration: AstId<ast::Module>,
        definition: FileId,
    },
    Inline {
        definition: AstId<ast::Module>,
    },
}

impl Default for ModuleOrigin {
    fn default() -> Self {
        ModuleOrigin::CrateRoot { definition: FileId(0) }
    }
}

impl ModuleOrigin {
    pub(crate) fn not_sure_file(file: Option<FileId>, declaration: AstId<ast::Module>) -> Self {
        match file {
            None => ModuleOrigin::Inline { definition: declaration },
            Some(definition) => ModuleOrigin::File { declaration, definition },
        }
    }

    fn declaration(&self) -> Option<AstId<ast::Module>> {
        match self {
            ModuleOrigin::File { declaration: module, .. }
            | ModuleOrigin::Inline { definition: module, .. } => Some(*module),
            ModuleOrigin::CrateRoot { .. } => None,
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

    /// Returns a node which defines this module.
    /// That is, a file or a `mod foo {}` with items.
    fn definition_source(&self, db: &impl DefDatabase) -> InFile<ModuleSource> {
        match self {
            ModuleOrigin::File { definition, .. } | ModuleOrigin::CrateRoot { definition } => {
                let file_id = *definition;
                let sf = db.parse(file_id).tree();
                return InFile::new(file_id.into(), ModuleSource::SourceFile(sf));
            }
            ModuleOrigin::Inline { definition } => {
                InFile::new(definition.file_id, ModuleSource::Module(definition.to_node(db)))
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

impl CrateDefMap {
    pub(crate) fn compute_crate_def_map_query(
        // Note that this doesn't have `+ AstDatabase`!
        // This gurantess that `CrateDefMap` is stable across reparses.
        db: &impl DefDatabase,
        krate: CrateId,
    ) -> Arc<CrateDefMap> {
        let _p = profile("compute_crate_def_map");
        let def_map = {
            let crate_graph = db.crate_graph();
            let edition = crate_graph.edition(krate);
            let mut modules: Arena<LocalModuleId, ModuleData> = Arena::default();
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
        let def_map = collector::collect_defs(db, def_map);
        Arc::new(def_map)
    }

    pub fn add_diagnostics(
        &self,
        db: &impl DefDatabase,
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

    pub(crate) fn resolve_path(
        &self,
        db: &impl DefDatabase,
        original_module: LocalModuleId,
        path: &ModPath,
        shadow: BuiltinShadowMode,
    ) -> (PerNs, Option<usize>) {
        let res =
            self.resolve_path_fp_with_macro(db, ResolveMode::Other, original_module, path, shadow);
        (res.resolved_def, res.segment_index)
    }
}

impl ModuleData {
    /// Returns a node which defines this module. That is, a file or a `mod foo {}` with items.
    pub fn definition_source(&self, db: &impl DefDatabase) -> InFile<ModuleSource> {
        self.origin.definition_source(db)
    }

    /// Returns a node which declares this module, either a `mod foo;` or a `mod foo {}`.
    /// `None` for the crate root or block.
    pub fn declaration_source(&self, db: &impl DefDatabase) -> Option<InFile<ast::Module>> {
        let decl = self.origin.declaration()?;
        let value = decl.to_node(db);
        Some(InFile { file_id: decl.file_id, value })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleSource {
    SourceFile(ast::SourceFile),
    Module(ast::Module),
}

impl ModuleSource {
    // FIXME: this methods do not belong here
    pub fn from_position(db: &impl DefDatabase, position: FilePosition) -> ModuleSource {
        let parse = db.parse(position.file_id);
        match &ra_syntax::algo::find_node_at_offset::<ast::Module>(
            parse.tree().syntax(),
            position.offset,
        ) {
            Some(m) if !m.has_semi() => ModuleSource::Module(m.clone()),
            _ => {
                let source_file = parse.tree();
                ModuleSource::SourceFile(source_file)
            }
        }
    }

    pub fn from_child_node(db: &impl DefDatabase, child: InFile<&SyntaxNode>) -> ModuleSource {
        if let Some(m) =
            child.value.ancestors().filter_map(ast::Module::cast).find(|it| !it.has_semi())
        {
            ModuleSource::Module(m)
        } else {
            let file_id = child.file_id.original_file(db);
            let source_file = db.parse(file_id).tree();
            ModuleSource::SourceFile(source_file)
        }
    }
}

mod diagnostics {
    use hir_expand::diagnostics::DiagnosticSink;
    use ra_db::RelativePathBuf;
    use ra_syntax::{ast, AstPtr};

    use crate::{db::DefDatabase, diagnostics::UnresolvedModule, nameres::LocalModuleId, AstId};

    #[derive(Debug, PartialEq, Eq)]
    pub(super) enum DefDiagnostic {
        UnresolvedModule {
            module: LocalModuleId,
            declaration: AstId<ast::Module>,
            candidate: RelativePathBuf,
        },
    }

    impl DefDiagnostic {
        pub(super) fn add_to(
            &self,
            db: &impl DefDatabase,
            target_module: LocalModuleId,
            sink: &mut DiagnosticSink,
        ) {
            match self {
                DefDiagnostic::UnresolvedModule { module, declaration, candidate } => {
                    if *module != target_module {
                        return;
                    }
                    let decl = declaration.to_node(db);
                    sink.push(UnresolvedModule {
                        file: declaration.file_id,
                        decl: AstPtr::new(&decl),
                        candidate: candidate.clone(),
                    })
                }
            }
        }
    }
}
