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

mod per_ns;
mod raw;
mod collector;
mod mod_resolution;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use once_cell::sync::Lazy;
use ra_arena::{impl_arena_id, Arena, RawId};
use ra_db::{Edition, FileId};
use ra_prof::profile;
use ra_syntax::ast;
use rustc_hash::{FxHashMap, FxHashSet};
use test_utils::tested_by;

use crate::{
    db::{AstDatabase, DefDatabase},
    diagnostics::DiagnosticSink,
    either::Either,
    ids::MacroDefId,
    nameres::diagnostics::DefDiagnostic,
    AstId, BuiltinType, Crate, HirFileId, MacroDef, Module, ModuleDef, Name, Path, PathKind, Trait,
};

pub(crate) use self::raw::{ImportSourceMap, RawItems};

pub use self::{
    per_ns::{Namespace, PerNs},
    raw::ImportId,
};

/// Contains all top-level defs from a macro-expanded crate
#[derive(Debug, PartialEq, Eq)]
pub struct CrateDefMap {
    krate: Crate,
    edition: Edition,
    /// The prelude module for this crate. This either comes from an import
    /// marked with the `prelude_import` attribute, or (in the normal case) from
    /// a dependency (`std` or `core`).
    prelude: Option<Module>,
    extern_prelude: FxHashMap<Name, ModuleDef>,
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

    exported_macros: FxHashMap<Name, MacroDefId>,

    diagnostics: Vec<DefDiagnostic>,
}

impl std::ops::Index<CrateModuleId> for CrateDefMap {
    type Output = ModuleData;
    fn index(&self, id: CrateModuleId) -> &ModuleData {
        &self.modules[id]
    }
}

/// An ID of a module, **local** to a specific crate
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CrateModuleId(RawId);
impl_arena_id!(CrateModuleId);

#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct ModuleData {
    pub(crate) parent: Option<CrateModuleId>,
    pub(crate) children: FxHashMap<Name, CrateModuleId>,
    pub(crate) scope: ModuleScope,
    /// None for root
    pub(crate) declaration: Option<AstId<ast::Module>>,
    /// None for inline modules.
    ///
    /// Note that non-inline modules, by definition, live inside non-macro file.
    pub(crate) definition: Option<FileId>,
}

#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct ModuleScope {
    items: FxHashMap<Name, Resolution>,
    /// Macros in current module scoped
    ///
    /// This scope works exactly the same way that item scoping does.
    /// Macro invocation with quantified path will search in it.
    /// See details below.
    macros: FxHashMap<Name, MacroDef>,
    /// Macros visable in current module in legacy textual scope
    ///
    /// For macros invoked by an unquatified identifier like `bar!()`, `legacy_macros` will be searched in first.
    /// If it yields no result, then it turns to module scoped `macros`.
    /// It macros with name quatified with a path like `crate::foo::bar!()`, `legacy_macros` will be skipped,
    /// and only normal scoped `macros` will be searched in.
    ///
    /// Note that this automatically inherit macros defined textually before the definition of module itself.
    legacy_macros: FxHashMap<Name, MacroDef>,
}

static BUILTIN_SCOPE: Lazy<FxHashMap<Name, Resolution>> = Lazy::new(|| {
    BuiltinType::ALL
        .iter()
        .map(|(name, ty)| {
            (name.clone(), Resolution { def: PerNs::types(ty.clone().into()), import: None })
        })
        .collect()
});

impl ModuleScope {
    pub fn entries<'a>(&'a self) -> impl Iterator<Item = (&'a Name, &'a Resolution)> + 'a {
        //FIXME: shadowing
        self.items.iter().chain(BUILTIN_SCOPE.iter())
    }
    pub fn get(&self, name: &Name) -> Option<&Resolution> {
        self.items.get(name).or_else(|| BUILTIN_SCOPE.get(name))
    }
    pub fn traits<'a>(&'a self) -> impl Iterator<Item = Trait> + 'a {
        self.items.values().filter_map(|r| match r.def.take_types() {
            Some(ModuleDef::Trait(t)) => Some(t),
            _ => None,
        })
    }
    /// It resolves in module scope. Textual scoped macros are ignored here.
    fn get_item_or_macro(&self, name: &Name) -> Option<ItemOrMacro> {
        match (self.get(name), self.macros.get(name)) {
            (Some(item), _) if !item.def.is_none() => Some(Either::A(item.def)),
            (_, Some(macro_)) => Some(Either::B(*macro_)),
            _ => None,
        }
    }
    fn get_legacy_macro(&self, name: &Name) -> Option<MacroDef> {
        self.legacy_macros.get(name).copied()
    }
}

type ItemOrMacro = Either<PerNs<ModuleDef>, MacroDef>;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Resolution {
    /// None for unresolved
    pub def: PerNs<ModuleDef>,
    /// ident by which this is imported into local scope.
    pub import: Option<ImportId>,
}

#[derive(Debug, Clone)]
struct ResolvePathResult {
    resolved_def: ItemOrMacro,
    segment_index: Option<usize>,
    reached_fixedpoint: ReachedFixedPoint,
}

impl ResolvePathResult {
    fn empty(reached_fixedpoint: ReachedFixedPoint) -> ResolvePathResult {
        ResolvePathResult::with(Either::A(PerNs::none()), reached_fixedpoint, None)
    }

    fn with(
        resolved_def: ItemOrMacro,
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

/// helper function for select item or macro to use
fn or(left: ItemOrMacro, right: ItemOrMacro) -> ItemOrMacro {
    match (left, right) {
        (Either::A(s), Either::A(o)) => Either::A(s.or(o)),
        (Either::B(s), _) => Either::B(s),
        (Either::A(s), Either::B(o)) => {
            if !s.is_none() {
                Either::A(s)
            } else {
                Either::B(o)
            }
        }
    }
}

impl CrateDefMap {
    pub(crate) fn crate_def_map_query(
        // Note that this doesn't have `+ AstDatabase`!
        // This gurantess that `CrateDefMap` is stable across reparses.
        db: &impl DefDatabase,
        krate: Crate,
    ) -> Arc<CrateDefMap> {
        let _p = profile("crate_def_map_query");
        let def_map = {
            let edition = krate.edition(db);
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
                exported_macros: FxHashMap::default(),
                diagnostics: Vec::new(),
            }
        };
        let def_map = collector::collect_defs(db, def_map);
        Arc::new(def_map)
    }

    pub(crate) fn krate(&self) -> Crate {
        self.krate
    }

    pub(crate) fn root(&self) -> CrateModuleId {
        self.root
    }

    pub(crate) fn mk_module(&self, module_id: CrateModuleId) -> Module {
        Module { krate: self.krate, module_id }
    }

    pub(crate) fn prelude(&self) -> Option<Module> {
        self.prelude
    }

    pub(crate) fn extern_prelude(&self) -> &FxHashMap<Name, ModuleDef> {
        &self.extern_prelude
    }

    pub(crate) fn add_diagnostics(
        &self,
        db: &(impl DefDatabase + AstDatabase),
        module: CrateModuleId,
        sink: &mut DiagnosticSink,
    ) {
        self.diagnostics.iter().for_each(|it| it.add_to(db, module, sink))
    }

    pub(crate) fn find_module_by_source(
        &self,
        file_id: HirFileId,
        decl_id: Option<AstId<ast::Module>>,
    ) -> Option<CrateModuleId> {
        let (module_id, _module_data) = self.modules.iter().find(|(_module_id, module_data)| {
            if decl_id.is_some() {
                module_data.declaration == decl_id
            } else {
                module_data.definition.map(|it| it.into()) == Some(file_id)
            }
        })?;
        Some(module_id)
    }

    pub(crate) fn resolve_path(
        &self,
        db: &impl DefDatabase,
        original_module: CrateModuleId,
        path: &Path,
    ) -> (PerNs<ModuleDef>, Option<usize>) {
        let res = self.resolve_path_fp_with_macro(db, ResolveMode::Other, original_module, path);
        (res.resolved_def.a().unwrap_or_else(PerNs::none), res.segment_index)
    }

    pub(crate) fn resolve_path_with_macro(
        &self,
        db: &impl DefDatabase,
        original_module: CrateModuleId,
        path: &Path,
    ) -> (ItemOrMacro, Option<usize>) {
        let res = self.resolve_path_fp_with_macro(db, ResolveMode::Other, original_module, path);
        (res.resolved_def, res.segment_index)
    }

    // Returns Yes if we are sure that additions to `ItemMap` wouldn't change
    // the result.
    fn resolve_path_fp_with_macro(
        &self,
        db: &impl DefDatabase,
        mode: ResolveMode,
        original_module: CrateModuleId,
        path: &Path,
    ) -> ResolvePathResult {
        let mut segments = path.segments.iter().enumerate();
        let mut curr_per_ns: ItemOrMacro = match path.kind {
            PathKind::Crate => {
                Either::A(PerNs::types(Module { krate: self.krate, module_id: self.root }.into()))
            }
            PathKind::Self_ => Either::A(PerNs::types(
                Module { krate: self.krate, module_id: original_module }.into(),
            )),
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
                self.resolve_name_in_module_with_macro(db, original_module, &segment.name)
            }
            PathKind::Super => {
                if let Some(p) = self.modules[original_module].parent {
                    Either::A(PerNs::types(Module { krate: self.krate, module_id: p }.into()))
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
                    Either::A(PerNs::types(*def))
                } else {
                    return ResolvePathResult::empty(ReachedFixedPoint::No); // extern crate declarations can add to the extern prelude
                }
            }
        };

        for (i, segment) in segments {
            let curr = match curr_per_ns.as_ref().a().and_then(|m| m.as_ref().take_types()) {
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
                ModuleDef::Module(module) => {
                    if module.krate != self.krate {
                        let path =
                            Path { segments: path.segments[i..].to_vec(), kind: PathKind::Self_ };
                        log::debug!("resolving {:?} in other crate", path);
                        let defp_map = db.crate_def_map(module.krate);
                        let (def, s) =
                            defp_map.resolve_path_with_macro(db, module.module_id, &path);
                        return ResolvePathResult::with(
                            def,
                            ReachedFixedPoint::Yes,
                            s.map(|s| s + i),
                        );
                    }

                    match self[module.module_id].scope.get_item_or_macro(&segment.name) {
                        Some(res) => res,
                        _ => {
                            log::debug!("path segment {:?} not found", segment.name);
                            return ResolvePathResult::empty(ReachedFixedPoint::No);
                        }
                    }
                }
                ModuleDef::Enum(e) => {
                    // enum variant
                    tested_by!(can_import_enum_variant);
                    match e.variant(db, &segment.name) {
                        Some(variant) => Either::A(PerNs::both(variant.into(), variant.into())),
                        None => {
                            return ResolvePathResult::with(
                                Either::A(PerNs::types((*e).into())),
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
                        Either::A(PerNs::types(*s)),
                        ReachedFixedPoint::Yes,
                        Some(i),
                    );
                }
            };
        }
        ResolvePathResult::with(curr_per_ns, ReachedFixedPoint::Yes, None)
    }

    fn resolve_name_in_crate_root_or_extern_prelude(&self, name: &Name) -> ItemOrMacro {
        let from_crate_root = self[self.root]
            .scope
            .get_item_or_macro(name)
            .unwrap_or_else(|| Either::A(PerNs::none()));
        let from_extern_prelude = self.resolve_name_in_extern_prelude(name);

        or(from_crate_root, Either::A(from_extern_prelude))
    }

    pub(crate) fn resolve_name_in_module(
        &self,
        db: &impl DefDatabase,
        module: CrateModuleId,
        name: &Name,
    ) -> PerNs<ModuleDef> {
        self.resolve_name_in_module_with_macro(db, module, name).a().unwrap_or_else(PerNs::none)
    }

    fn resolve_name_in_module_with_macro(
        &self,
        db: &impl DefDatabase,
        module: CrateModuleId,
        name: &Name,
    ) -> ItemOrMacro {
        // Resolve in:
        //  - legacy scope
        //  - current module / scope
        //  - extern prelude
        //  - std prelude
        let from_legacy_macro = self[module]
            .scope
            .get_legacy_macro(name)
            .map_or_else(|| Either::A(PerNs::none()), Either::B);
        let from_scope =
            self[module].scope.get_item_or_macro(name).unwrap_or_else(|| Either::A(PerNs::none()));
        let from_extern_prelude =
            self.extern_prelude.get(name).map_or(PerNs::none(), |&it| PerNs::types(it));
        let from_prelude = self.resolve_in_prelude(db, name);

        or(from_legacy_macro, or(from_scope, or(Either::A(from_extern_prelude), from_prelude)))
    }

    fn resolve_name_in_extern_prelude(&self, name: &Name) -> PerNs<ModuleDef> {
        self.extern_prelude.get(name).map_or(PerNs::none(), |&it| PerNs::types(it))
    }

    fn resolve_in_prelude(&self, db: &impl DefDatabase, name: &Name) -> ItemOrMacro {
        if let Some(prelude) = self.prelude {
            let resolution = if prelude.krate == self.krate {
                self[prelude.module_id].scope.get_item_or_macro(name)
            } else {
                db.crate_def_map(prelude.krate)[prelude.module_id].scope.get_item_or_macro(name)
            };
            resolution.unwrap_or_else(|| Either::A(PerNs::none()))
        } else {
            Either::A(PerNs::none())
        }
    }
}

mod diagnostics {
    use ra_syntax::{ast, AstPtr};
    use relative_path::RelativePathBuf;

    use crate::{
        db::{AstDatabase, DefDatabase},
        diagnostics::{DiagnosticSink, UnresolvedModule},
        nameres::CrateModuleId,
        AstId,
    };

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
            db: &(impl DefDatabase + AstDatabase),
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
