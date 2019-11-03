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
//!  This happens in the `raw` module, which parses a single source file into a
//!  set of top-level items. Nested imports are desugared to flat imports in
//!  this phase. Macro calls are represented as a triple of (Path, Option<Name>,
//!  TokenTree).
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
//! path and, upon success, we run macro expansion and "collect module" phase
//! on the result

// FIXME: review privacy of submodules
pub mod raw;
pub mod per_ns;
pub mod collector;
pub mod mod_resolution;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use hir_expand::{diagnostics::DiagnosticSink, name::Name, MacroDefId};
use once_cell::sync::Lazy;
use ra_arena::Arena;
use ra_db::{CrateId, Edition, FileId};
use ra_prof::profile;
use ra_syntax::ast;
use rustc_hash::{FxHashMap, FxHashSet};
use test_utils::tested_by;

use crate::{
    builtin_type::BuiltinType,
    db::DefDatabase2,
    nameres::{diagnostics::DefDiagnostic, per_ns::PerNs, raw::ImportId},
    path::{Path, PathKind},
    AdtId, AstId, CrateModuleId, EnumVariantId, ModuleDefId, ModuleId, TraitId,
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
    pub modules: Arena<CrateModuleId, ModuleData>,

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
}

#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct ModuleScope {
    pub items: FxHashMap<Name, Resolution>,
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

impl Resolution {
    pub(crate) fn from_macro(macro_: MacroDefId) -> Self {
        Resolution { def: PerNs::macros(macro_), import: None }
    }
}

#[derive(Debug, Clone)]
struct ResolvePathResult {
    resolved_def: PerNs,
    segment_index: Option<usize>,
    reached_fixedpoint: ReachedFixedPoint,
}

impl ResolvePathResult {
    fn empty(reached_fixedpoint: ReachedFixedPoint) -> ResolvePathResult {
        ResolvePathResult::with(PerNs::none(), reached_fixedpoint, None)
    }

    fn with(
        resolved_def: PerNs,
        reached_fixedpoint: ReachedFixedPoint,
        segment_index: Option<usize>,
    ) -> ResolvePathResult {
        ResolvePathResult { resolved_def, reached_fixedpoint, segment_index }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolveMode {
    Import,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReachedFixedPoint {
    Yes,
    No,
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

    // Returns Yes if we are sure that additions to `ItemMap` wouldn't change
    // the result.
    fn resolve_path_fp_with_macro(
        &self,
        db: &impl DefDatabase2,
        mode: ResolveMode,
        original_module: CrateModuleId,
        path: &Path,
    ) -> ResolvePathResult {
        let mut segments = path.segments.iter().enumerate();
        let mut curr_per_ns: PerNs = match path.kind {
            PathKind::DollarCrate(krate) => {
                if krate == self.krate {
                    tested_by!(macro_dollar_crate_self);
                    PerNs::types(ModuleId { krate: self.krate, module_id: self.root }.into())
                } else {
                    let def_map = db.crate_def_map(krate);
                    let module = ModuleId { krate, module_id: def_map.root };
                    tested_by!(macro_dollar_crate_other);
                    PerNs::types(module.into())
                }
            }
            PathKind::Crate => {
                PerNs::types(ModuleId { krate: self.krate, module_id: self.root }.into())
            }
            PathKind::Self_ => {
                PerNs::types(ModuleId { krate: self.krate, module_id: original_module }.into())
            }
            // plain import or absolute path in 2015: crate-relative with
            // fallback to extern prelude (with the simplification in
            // rust-lang/rust#57745)
            // FIXME there must be a nicer way to write this condition
            PathKind::Plain | PathKind::Abs
                if self.edition == Edition::Edition2015
                    && (path.kind == PathKind::Abs || mode == ResolveMode::Import) =>
            {
                let segment = match segments.next() {
                    Some((_, segment)) => segment,
                    None => return ResolvePathResult::empty(ReachedFixedPoint::Yes),
                };
                log::debug!("resolving {:?} in crate root (+ extern prelude)", segment);
                self.resolve_name_in_crate_root_or_extern_prelude(&segment.name)
            }
            PathKind::Plain => {
                let segment = match segments.next() {
                    Some((_, segment)) => segment,
                    None => return ResolvePathResult::empty(ReachedFixedPoint::Yes),
                };
                log::debug!("resolving {:?} in module", segment);
                self.resolve_name_in_module(db, original_module, &segment.name)
            }
            PathKind::Super => {
                if let Some(p) = self.modules[original_module].parent {
                    PerNs::types(ModuleId { krate: self.krate, module_id: p }.into())
                } else {
                    log::debug!("super path in root module");
                    return ResolvePathResult::empty(ReachedFixedPoint::Yes);
                }
            }
            PathKind::Abs => {
                // 2018-style absolute path -- only extern prelude
                let segment = match segments.next() {
                    Some((_, segment)) => segment,
                    None => return ResolvePathResult::empty(ReachedFixedPoint::Yes),
                };
                if let Some(def) = self.extern_prelude.get(&segment.name) {
                    log::debug!("absolute path {:?} resolved to crate {:?}", path, def);
                    PerNs::types(*def)
                } else {
                    return ResolvePathResult::empty(ReachedFixedPoint::No); // extern crate declarations can add to the extern prelude
                }
            }
            PathKind::Type(_) => {
                // This is handled in `infer::infer_path_expr`
                // The result returned here does not matter
                return ResolvePathResult::empty(ReachedFixedPoint::Yes);
            }
        };

        for (i, segment) in segments {
            let curr = match curr_per_ns.take_types() {
                Some(r) => r,
                None => {
                    // we still have path segments left, but the path so far
                    // didn't resolve in the types namespace => no resolution
                    // (don't break here because `curr_per_ns` might contain
                    // something in the value namespace, and it would be wrong
                    // to return that)
                    return ResolvePathResult::empty(ReachedFixedPoint::No);
                }
            };
            // resolve segment in curr

            curr_per_ns = match curr {
                ModuleDefId::ModuleId(module) => {
                    if module.krate != self.krate {
                        let path =
                            Path { segments: path.segments[i..].to_vec(), kind: PathKind::Self_ };
                        log::debug!("resolving {:?} in other crate", path);
                        let defp_map = db.crate_def_map(module.krate);
                        let (def, s) = defp_map.resolve_path(db, module.module_id, &path);
                        return ResolvePathResult::with(
                            def,
                            ReachedFixedPoint::Yes,
                            s.map(|s| s + i),
                        );
                    }

                    // Since it is a qualified path here, it should not contains legacy macros
                    match self[module.module_id].scope.get(&segment.name) {
                        Some(res) => res.def,
                        _ => {
                            log::debug!("path segment {:?} not found", segment.name);
                            return ResolvePathResult::empty(ReachedFixedPoint::No);
                        }
                    }
                }
                ModuleDefId::AdtId(AdtId::EnumId(e)) => {
                    // enum variant
                    tested_by!(can_import_enum_variant);
                    let enum_data = db.enum_data(e);
                    match enum_data.variant(&segment.name) {
                        Some(local_id) => {
                            let variant = EnumVariantId { parent: e, local_id };
                            PerNs::both(variant.into(), variant.into())
                        }
                        None => {
                            return ResolvePathResult::with(
                                PerNs::types(e.into()),
                                ReachedFixedPoint::Yes,
                                Some(i),
                            );
                        }
                    }
                }
                s => {
                    // could be an inherent method call in UFCS form
                    // (`Struct::method`), or some other kind of associated item
                    log::debug!(
                        "path segment {:?} resolved to non-module {:?}, but is not last",
                        segment.name,
                        curr,
                    );

                    return ResolvePathResult::with(
                        PerNs::types(s),
                        ReachedFixedPoint::Yes,
                        Some(i),
                    );
                }
            };
        }
        ResolvePathResult::with(curr_per_ns, ReachedFixedPoint::Yes, None)
    }

    fn resolve_name_in_crate_root_or_extern_prelude(&self, name: &Name) -> PerNs {
        let from_crate_root =
            self[self.root].scope.get(name).map_or_else(PerNs::none, |res| res.def);
        let from_extern_prelude = self.resolve_name_in_extern_prelude(name);

        from_crate_root.or(from_extern_prelude)
    }

    pub(crate) fn resolve_name_in_module(
        &self,
        db: &impl DefDatabase2,
        module: CrateModuleId,
        name: &Name,
    ) -> PerNs {
        // Resolve in:
        //  - legacy scope of macro
        //  - current module / scope
        //  - extern prelude
        //  - std prelude
        let from_legacy_macro =
            self[module].scope.get_legacy_macro(name).map_or_else(PerNs::none, PerNs::macros);
        let from_scope = self[module].scope.get(name).map_or_else(PerNs::none, |res| res.def);
        let from_extern_prelude =
            self.extern_prelude.get(name).map_or(PerNs::none(), |&it| PerNs::types(it));
        let from_prelude = self.resolve_in_prelude(db, name);

        from_legacy_macro.or(from_scope).or(from_extern_prelude).or(from_prelude)
    }

    fn resolve_name_in_extern_prelude(&self, name: &Name) -> PerNs {
        self.extern_prelude.get(name).map_or(PerNs::none(), |&it| PerNs::types(it))
    }

    fn resolve_in_prelude(&self, db: &impl DefDatabase2, name: &Name) -> PerNs {
        if let Some(prelude) = self.prelude {
            let keep;
            let def_map = if prelude.krate == self.krate {
                self
            } else {
                // Extend lifetime
                keep = db.crate_def_map(prelude.krate);
                &keep
            };
            def_map[prelude.module_id].scope.get(name).map_or_else(PerNs::none, |res| res.def)
        } else {
            PerNs::none()
        }
    }
}

mod diagnostics {
    use hir_expand::diagnostics::DiagnosticSink;
    use ra_syntax::{ast, AstPtr};
    use relative_path::RelativePathBuf;

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
