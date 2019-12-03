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

use either::Either;
use hir_expand::{
    ast_id_map::FileAstId, diagnostics::DiagnosticSink, name::Name, InFile, MacroDefId,
};
use once_cell::sync::Lazy;
use ra_arena::Arena;
use ra_db::{CrateId, Edition, FileId};
use ra_prof::profile;
use ra_syntax::ast;
use rustc_hash::FxHashMap;

use crate::{
    builtin_type::BuiltinType,
    db::DefDatabase,
    nameres::{diagnostics::DefDiagnostic, path_resolution::ResolveMode},
    path::Path,
    per_ns::PerNs,
    AstId, FunctionId, ImplId, LocalImportId, LocalModuleId, ModuleDefId, ModuleId, TraitId,
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
    /// It should not be `None` after collecting definitions.
    Root(Option<FileId>),
    /// Note that non-inline modules, by definition, live inside non-macro file.
    File(AstId<ast::Module>, FileId),
    Inline(AstId<ast::Module>),
    Block(AstId<ast::Block>),
}

impl Default for ModuleOrigin {
    fn default() -> Self {
        ModuleOrigin::Root(None)
    }
}

impl ModuleOrigin {
    pub fn root(file_id: FileId) -> Self {
        ModuleOrigin::Root(Some(file_id))
    }

    pub fn not_sure_file(file: Option<FileId>, module: AstId<ast::Module>) -> Self {
        match file {
            None => ModuleOrigin::Inline(module),
            Some(file) => ModuleOrigin::File(module, file),
        }
    }

    pub fn not_sure_mod(file: FileId, module: Option<AstId<ast::Module>>) -> Self {
        match module {
            None => ModuleOrigin::root(file),
            Some(module) => ModuleOrigin::File(module, file),
        }
    }

    pub fn declaration(&self) -> Option<AstId<ast::Module>> {
        match self {
            ModuleOrigin::File(m, _) | ModuleOrigin::Inline(m) => Some(*m),
            ModuleOrigin::Root(_) | ModuleOrigin::Block(_) => None,
        }
    }

    pub fn file_id(&self) -> Option<FileId> {
        match self {
            ModuleOrigin::File(_, file_id) | ModuleOrigin::Root(Some(file_id)) => Some(*file_id),
            _ => None,
        }
    }

    /// Returns a node which defines this module.
    /// That is, a file or a `mod foo {}` with items.
    pub fn definition_source(
        &self,
        db: &impl DefDatabase,
    ) -> InFile<Either<ast::SourceFile, ast::Module>> {
        match self {
            ModuleOrigin::File(_, file_id) | ModuleOrigin::Root(Some(file_id)) => {
                let file_id = *file_id;
                let sf = db.parse(file_id).tree();
                return InFile::new(file_id.into(), Either::Left(sf));
            }
            ModuleOrigin::Root(None) => unreachable!(),
            ModuleOrigin::Inline(m) => InFile::new(m.file_id, Either::Right(m.to_node(db))),
            // FIXME: right now it's never constructed, so it's fine to omit
            ModuleOrigin::Block(_b) => unimplemented!(),
        }
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
pub struct ModuleData {
    pub parent: Option<LocalModuleId>,
    pub children: FxHashMap<Name, LocalModuleId>,
    pub scope: ModuleScope,

    /// Where does this module come from?
    pub origin: ModuleOrigin,

    pub impls: Vec<ImplId>,
}

#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct Declarations {
    fns: FxHashMap<FileAstId<ast::FnDef>, FunctionId>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ModuleScope {
    items: FxHashMap<Name, Resolution>,
    /// Macros visable in current module in legacy textual scope
    ///
    /// For macros invoked by an unquatified identifier like `bar!()`, `legacy_macros` will be searched in first.
    /// If it yields no result, then it turns to module scoped `macros`.
    /// It macros with name quatified with a path like `crate::foo::bar!()`, `legacy_macros` will be skipped,
    /// and only normal scoped `macros` will be searched in.
    ///
    /// Note that this automatically inherit macros defined textually before the definition of module itself.
    ///
    /// Module scoped macros will be inserted into `items` instead of here.
    // FIXME: Macro shadowing in one module is not properly handled. Non-item place macros will
    // be all resolved to the last one defined if shadowing happens.
    legacy_macros: FxHashMap<Name, MacroDefId>,
}

static BUILTIN_SCOPE: Lazy<FxHashMap<Name, Resolution>> = Lazy::new(|| {
    BuiltinType::ALL
        .iter()
        .map(|(name, ty)| {
            (name.clone(), Resolution { def: PerNs::types(ty.clone().into()), import: None })
        })
        .collect()
});

/// Shadow mode for builtin type which can be shadowed by module.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BuiltinShadowMode {
    // Prefer Module
    Module,
    // Prefer Other Types
    Other,
}

/// Legacy macros can only be accessed through special methods like `get_legacy_macros`.
/// Other methods will only resolve values, types and module scoped macros only.
impl ModuleScope {
    pub fn entries<'a>(&'a self) -> impl Iterator<Item = (&'a Name, &'a Resolution)> + 'a {
        //FIXME: shadowing
        self.items.iter().chain(BUILTIN_SCOPE.iter())
    }

    pub fn declarations(&self) -> impl Iterator<Item = ModuleDefId> + '_ {
        self.entries()
            .filter_map(|(_name, res)| if res.import.is_none() { Some(res.def) } else { None })
            .flat_map(|per_ns| {
                per_ns.take_types().into_iter().chain(per_ns.take_values().into_iter())
            })
    }

    /// Iterate over all module scoped macros
    pub fn macros<'a>(&'a self) -> impl Iterator<Item = (&'a Name, MacroDefId)> + 'a {
        self.items
            .iter()
            .filter_map(|(name, res)| res.def.take_macros().map(|macro_| (name, macro_)))
    }

    /// Iterate over all legacy textual scoped macros visable at the end of the module
    pub fn legacy_macros<'a>(&'a self) -> impl Iterator<Item = (&'a Name, MacroDefId)> + 'a {
        self.legacy_macros.iter().map(|(name, def)| (name, *def))
    }

    /// Get a name from current module scope, legacy macros are not included
    pub fn get(&self, name: &Name, shadow: BuiltinShadowMode) -> Option<&Resolution> {
        match shadow {
            BuiltinShadowMode::Module => self.items.get(name).or_else(|| BUILTIN_SCOPE.get(name)),
            BuiltinShadowMode::Other => {
                let item = self.items.get(name);
                if let Some(res) = item {
                    if let Some(ModuleDefId::ModuleId(_)) = res.def.take_types() {
                        return BUILTIN_SCOPE.get(name).or(item);
                    }
                }

                item.or_else(|| BUILTIN_SCOPE.get(name))
            }
        }
    }

    pub fn traits<'a>(&'a self) -> impl Iterator<Item = TraitId> + 'a {
        self.items.values().filter_map(|r| match r.def.take_types() {
            Some(ModuleDefId::TraitId(t)) => Some(t),
            _ => None,
        })
    }

    fn get_legacy_macro(&self, name: &Name) -> Option<MacroDefId> {
        self.legacy_macros.get(name).copied()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Resolution {
    /// None for unresolved
    pub def: PerNs,
    /// ident by which this is imported into local scope.
    pub import: Option<LocalImportId>,
}

impl CrateDefMap {
    pub(crate) fn crate_def_map_query(
        // Note that this doesn't have `+ AstDatabase`!
        // This gurantess that `CrateDefMap` is stable across reparses.
        db: &impl DefDatabase,
        krate: CrateId,
    ) -> Arc<CrateDefMap> {
        let _p = profile("crate_def_map_query");
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
        path: &Path,
        shadow: BuiltinShadowMode,
    ) -> (PerNs, Option<usize>) {
        let res =
            self.resolve_path_fp_with_macro(db, ResolveMode::Other, original_module, path, shadow);
        (res.resolved_def, res.segment_index)
    }
}

impl ModuleData {
    /// Returns a node which defines this module. That is, a file or a `mod foo {}` with items.
    pub fn definition_source(
        &self,
        db: &impl DefDatabase,
    ) -> InFile<Either<ast::SourceFile, ast::Module>> {
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
