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

pub mod raw;
pub mod per_ns;
mod collector;
mod mod_resolution;
mod path_resolution;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use hir_expand::{ast_id_map::FileAstId, diagnostics::DiagnosticSink, name::Name, MacroDefId};
use once_cell::sync::Lazy;
use ra_arena::Arena;
use ra_db::{CrateId, Edition, FileId};
use ra_prof::profile;
use ra_syntax::ast;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    builtin_type::BuiltinType,
    db::DefDatabase2,
    nameres::{
        diagnostics::DefDiagnostic, path_resolution::ResolveMode, per_ns::PerNs, raw::ImportId,
    },
    path::Path,
    AstId, CrateModuleId, FunctionId, ImplId, ModuleDefId, ModuleId, TraitId,
};

/// Contains all top-level defs from a macro-expanded crate
#[derive(Debug, PartialEq, Eq)]
pub struct CrateDefMap {
    krate: CrateId,
    edition: Edition,
    /// The prelude module for this crate. This either comes from an import
    /// marked with the `prelude_import` attribute, or (in the normal case) from
    /// a dependency (`std` or `core`).
    prelude: Option<ModuleId>,
    extern_prelude: FxHashMap<Name, ModuleDefId>,
    root: CrateModuleId,
    modules: Arena<CrateModuleId, ModuleData>,

    /// Some macros are not well-behavior, which leads to infinite loop
    /// e.g. macro_rules! foo { ($ty:ty) => { foo!($ty); } }
    /// We mark it down and skip it in collector
    ///
    /// FIXME:
    /// Right now it only handle a poison macro in a single crate,
    /// such that if other crate try to call that macro,
    /// the whole process will do again until it became poisoned in that crate.
    /// We should handle this macro set globally
    /// However, do we want to put it as a global variable?
    poison_macros: FxHashSet<MacroDefId>,

    diagnostics: Vec<DefDiagnostic>,
}

impl std::ops::Index<CrateModuleId> for CrateDefMap {
    type Output = ModuleData;
    fn index(&self, id: CrateModuleId) -> &ModuleData {
        &self.modules[id]
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
pub struct ModuleData {
    pub parent: Option<CrateModuleId>,
    pub children: FxHashMap<Name, CrateModuleId>,
    pub scope: ModuleScope,
    /// None for root
    pub declaration: Option<AstId<ast::Module>>,
    /// None for inline modules.
    ///
    /// Note that non-inline modules, by definition, live inside non-macro file.
    pub definition: Option<FileId>,
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

/// Legacy macros can only be accessed through special methods like `get_legacy_macros`.
/// Other methods will only resolve values, types and module scoped macros only.
impl ModuleScope {
    pub fn entries<'a>(&'a self) -> impl Iterator<Item = (&'a Name, &'a Resolution)> + 'a {
        //FIXME: shadowing
        self.items.iter().chain(BUILTIN_SCOPE.iter())
    }

    /// Iterate over all module scoped macros
    pub fn macros<'a>(&'a self) -> impl Iterator<Item = (&'a Name, MacroDefId)> + 'a {
        self.items
            .iter()
            .filter_map(|(name, res)| res.def.get_macros().map(|macro_| (name, macro_)))
    }

    /// Iterate over all legacy textual scoped macros visable at the end of the module
    pub fn legacy_macros<'a>(&'a self) -> impl Iterator<Item = (&'a Name, MacroDefId)> + 'a {
        self.legacy_macros.iter().map(|(name, def)| (name, *def))
    }

    /// Get a name from current module scope, legacy macros are not included
    pub fn get(&self, name: &Name) -> Option<&Resolution> {
        self.items.get(name).or_else(|| BUILTIN_SCOPE.get(name))
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
    pub import: Option<ImportId>,
}

impl CrateDefMap {
    pub(crate) fn crate_def_map_query(
        // Note that this doesn't have `+ AstDatabase`!
        // This gurantess that `CrateDefMap` is stable across reparses.
        db: &impl DefDatabase2,
        krate: CrateId,
    ) -> Arc<CrateDefMap> {
        let _p = profile("crate_def_map_query");
        let def_map = {
            let crate_graph = db.crate_graph();
            let edition = crate_graph.edition(krate);
            let mut modules: Arena<CrateModuleId, ModuleData> = Arena::default();
            let root = modules.alloc(ModuleData::default());
            CrateDefMap {
                krate,
                edition,
                extern_prelude: FxHashMap::default(),
                prelude: None,
                root,
                modules,
                poison_macros: FxHashSet::default(),
                diagnostics: Vec::new(),
            }
        };
        let def_map = collector::collect_defs(db, def_map);
        Arc::new(def_map)
    }

    pub fn krate(&self) -> CrateId {
        self.krate
    }

    pub fn root(&self) -> CrateModuleId {
        self.root
    }

    pub fn prelude(&self) -> Option<ModuleId> {
        self.prelude
    }

    pub fn extern_prelude(&self) -> &FxHashMap<Name, ModuleDefId> {
        &self.extern_prelude
    }

    pub fn add_diagnostics(
        &self,
        db: &impl DefDatabase2,
        module: CrateModuleId,
        sink: &mut DiagnosticSink,
    ) {
        self.diagnostics.iter().for_each(|it| it.add_to(db, module, sink))
    }

    pub fn resolve_path(
        &self,
        db: &impl DefDatabase2,
        original_module: CrateModuleId,
        path: &Path,
    ) -> (PerNs, Option<usize>) {
        let res = self.resolve_path_fp_with_macro(db, ResolveMode::Other, original_module, path);
        (res.resolved_def, res.segment_index)
    }

    pub fn modules(&self) -> impl Iterator<Item = CrateModuleId> + '_ {
        self.modules.iter().map(|(id, _data)| id)
    }

    pub fn modules_for_file(&self, file_id: FileId) -> impl Iterator<Item = CrateModuleId> + '_ {
        self.modules
            .iter()
            .filter(move |(_id, data)| data.definition == Some(file_id))
            .map(|(id, _data)| id)
    }
}

mod diagnostics {
    use hir_expand::diagnostics::DiagnosticSink;
    use ra_db::RelativePathBuf;
    use ra_syntax::{ast, AstPtr};

    use crate::{db::DefDatabase2, diagnostics::UnresolvedModule, nameres::CrateModuleId, AstId};

    #[derive(Debug, PartialEq, Eq)]
    pub(super) enum DefDiagnostic {
        UnresolvedModule {
            module: CrateModuleId,
            declaration: AstId<ast::Module>,
            candidate: RelativePathBuf,
        },
    }

    impl DefDiagnostic {
        pub(super) fn add_to(
            &self,
            db: &impl DefDatabase2,
            target_module: CrateModuleId,
            sink: &mut DiagnosticSink,
        ) {
            match self {
                DefDiagnostic::UnresolvedModule { module, declaration, candidate } => {
                    if *module != target_module {
                        return;
                    }
                    let decl = declaration.to_node(db);
                    sink.push(UnresolvedModule {
                        file: declaration.file_id(),
                        decl: AstPtr::new(&decl),
                        candidate: candidate.clone(),
                    })
                }
            }
        }
    }
}
