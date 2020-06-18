//! Defines a unit of change that can applied to a state of IDE to get the next
//! state. Changes are transactional.

use std::{fmt, sync::Arc, time};

use ra_db::{
    salsa::{Database, Durability, SweepStrategy},
    CrateGraph, FileId, RelativePathBuf, SourceDatabase, SourceDatabaseExt, SourceRoot,
    SourceRootId,
};
use ra_prof::{memory_usage, profile, Bytes};
use rustc_hash::FxHashMap;

use crate::{symbol_index::SymbolsDatabase, RootDatabase};

#[derive(Default)]
pub struct AnalysisChange {
    new_roots: Vec<(SourceRootId, bool)>,
    roots_changed: FxHashMap<SourceRootId, RootChange>,
    files_changed: Vec<(FileId, Arc<String>)>,
    crate_graph: Option<CrateGraph>,
}

impl fmt::Debug for AnalysisChange {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut d = fmt.debug_struct("AnalysisChange");
        if !self.new_roots.is_empty() {
            d.field("new_roots", &self.new_roots);
        }
        if !self.roots_changed.is_empty() {
            d.field("roots_changed", &self.roots_changed);
        }
        if !self.files_changed.is_empty() {
            d.field("files_changed", &self.files_changed.len());
        }
        if self.crate_graph.is_some() {
            d.field("crate_graph", &self.crate_graph);
        }
        d.finish()
    }
}

impl AnalysisChange {
    pub fn new() -> AnalysisChange {
        AnalysisChange::default()
    }

    pub fn add_root(&mut self, root_id: SourceRootId, is_local: bool) {
        self.new_roots.push((root_id, is_local));
    }

    pub fn add_file(
        &mut self,
        root_id: SourceRootId,
        file_id: FileId,
        path: RelativePathBuf,
        text: Arc<String>,
    ) {
        let file = AddFile { file_id, path, text };
        self.roots_changed.entry(root_id).or_default().added.push(file);
    }

    pub fn change_file(&mut self, file_id: FileId, new_text: Arc<String>) {
        self.files_changed.push((file_id, new_text))
    }

    pub fn remove_file(&mut self, root_id: SourceRootId, file_id: FileId, path: RelativePathBuf) {
        let file = RemoveFile { file_id, path };
        self.roots_changed.entry(root_id).or_default().removed.push(file);
    }

    pub fn set_crate_graph(&mut self, graph: CrateGraph) {
        self.crate_graph = Some(graph);
    }
}

#[derive(Debug)]
struct AddFile {
    file_id: FileId,
    path: RelativePathBuf,
    text: Arc<String>,
}

#[derive(Debug)]
struct RemoveFile {
    file_id: FileId,
    path: RelativePathBuf,
}

#[derive(Default)]
struct RootChange {
    added: Vec<AddFile>,
    removed: Vec<RemoveFile>,
}

impl fmt::Debug for RootChange {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("AnalysisChange")
            .field("added", &self.added.len())
            .field("removed", &self.removed.len())
            .finish()
    }
}

const GC_COOLDOWN: time::Duration = time::Duration::from_millis(100);

impl RootDatabase {
    pub fn request_cancellation(&mut self) {
        let _p = profile("RootDatabase::request_cancellation");
        self.salsa_runtime_mut().synthetic_write(Durability::LOW);
    }

    pub fn apply_change(&mut self, change: AnalysisChange) {
        let _p = profile("RootDatabase::apply_change");
        self.request_cancellation();
        log::info!("apply_change {:?}", change);
        if !change.new_roots.is_empty() {
            let mut local_roots = Vec::clone(&self.local_roots());
            let mut libraries = Vec::clone(&self.library_roots());
            for (root_id, is_local) in change.new_roots {
                let root =
                    if is_local { SourceRoot::new_local() } else { SourceRoot::new_library() };
                let durability = durability(&root);
                self.set_source_root_with_durability(root_id, Arc::new(root), durability);
                if is_local {
                    local_roots.push(root_id);
                } else {
                    libraries.push(root_id)
                }
            }
            self.set_local_roots_with_durability(Arc::new(local_roots), Durability::HIGH);
            self.set_library_roots_with_durability(Arc::new(libraries), Durability::HIGH);
        }

        for (root_id, root_change) in change.roots_changed {
            self.apply_root_change(root_id, root_change);
        }
        for (file_id, text) in change.files_changed {
            let source_root_id = self.file_source_root(file_id);
            let source_root = self.source_root(source_root_id);
            let durability = durability(&source_root);
            self.set_file_text_with_durability(file_id, text, durability)
        }
        if let Some(crate_graph) = change.crate_graph {
            self.set_crate_graph_with_durability(Arc::new(crate_graph), Durability::HIGH)
        }
    }

    fn apply_root_change(&mut self, root_id: SourceRootId, root_change: RootChange) {
        let mut source_root = SourceRoot::clone(&self.source_root(root_id));
        let durability = durability(&source_root);
        for add_file in root_change.added {
            self.set_file_text_with_durability(add_file.file_id, add_file.text, durability);
            self.set_file_relative_path_with_durability(
                add_file.file_id,
                add_file.path.clone(),
                durability,
            );
            self.set_file_source_root_with_durability(add_file.file_id, root_id, durability);
            source_root.insert_file(add_file.path, add_file.file_id);
        }
        for remove_file in root_change.removed {
            self.set_file_text_with_durability(remove_file.file_id, Default::default(), durability);
            source_root.remove_file(&remove_file.path);
        }
        self.set_source_root_with_durability(root_id, Arc::new(source_root), durability);
    }

    pub fn maybe_collect_garbage(&mut self) {
        if cfg!(feature = "wasm") {
            return;
        }

        if self.last_gc_check.elapsed() > GC_COOLDOWN {
            self.last_gc_check = crate::wasm_shims::Instant::now();
        }
    }

    pub fn collect_garbage(&mut self) {
        if cfg!(feature = "wasm") {
            return;
        }

        let _p = profile("RootDatabase::collect_garbage");
        self.last_gc = crate::wasm_shims::Instant::now();

        let sweep = SweepStrategy::default().discard_values().sweep_all_revisions();

        self.query(ra_db::ParseQuery).sweep(sweep);
        self.query(hir::db::ParseMacroQuery).sweep(sweep);

        // Macros do take significant space, but less then the syntax trees
        // self.query(hir::db::MacroDefQuery).sweep(sweep);
        // self.query(hir::db::MacroArgQuery).sweep(sweep);
        // self.query(hir::db::MacroExpandQuery).sweep(sweep);

        self.query(hir::db::AstIdMapQuery).sweep(sweep);

        self.query(hir::db::BodyWithSourceMapQuery).sweep(sweep);

        self.query(hir::db::ExprScopesQuery).sweep(sweep);
        self.query(hir::db::InferQueryQuery).sweep(sweep);
        self.query(hir::db::BodyQuery).sweep(sweep);
    }

    pub fn per_query_memory_usage(&mut self) -> Vec<(String, Bytes)> {
        let mut acc: Vec<(String, Bytes)> = vec![];
        let sweep = SweepStrategy::default().discard_values().sweep_all_revisions();
        macro_rules! sweep_each_query {
            ($($q:path)*) => {$(
                let before = memory_usage().allocated;
                self.query($q).sweep(sweep);
                let after = memory_usage().allocated;
                let q: $q = Default::default();
                let name = format!("{:?}", q);
                acc.push((name, before - after));

                let before = memory_usage().allocated;
                self.query($q).sweep(sweep.discard_everything());
                let after = memory_usage().allocated;
                let q: $q = Default::default();
                let name = format!("{:?} (deps)", q);
                acc.push((name, before - after));
            )*}
        }
        sweep_each_query![
            // SourceDatabase
            ra_db::ParseQuery
            ra_db::SourceRootCratesQuery

            // AstDatabase
            hir::db::AstIdMapQuery
            hir::db::InternMacroQuery
            hir::db::MacroArgQuery
            hir::db::MacroDefQuery
            hir::db::ParseMacroQuery
            hir::db::MacroExpandQuery
            hir::db::InternEagerExpansionQuery

            // DefDatabase
            hir::db::RawItemsQuery
            hir::db::CrateDefMapQueryQuery
            hir::db::StructDataQuery
            hir::db::UnionDataQuery
            hir::db::EnumDataQuery
            hir::db::ImplDataQuery
            hir::db::TraitDataQuery
            hir::db::TypeAliasDataQuery
            hir::db::FunctionDataQuery
            hir::db::ConstDataQuery
            hir::db::StaticDataQuery
            hir::db::BodyWithSourceMapQuery
            hir::db::BodyQuery
            hir::db::ExprScopesQuery
            hir::db::GenericParamsQuery
            hir::db::AttrsQuery
            hir::db::ModuleLangItemsQuery
            hir::db::CrateLangItemsQuery
            hir::db::LangItemQuery
            hir::db::DocumentationQuery
            hir::db::ImportMapQuery

            // InternDatabase
            hir::db::InternFunctionQuery
            hir::db::InternStructQuery
            hir::db::InternUnionQuery
            hir::db::InternEnumQuery
            hir::db::InternConstQuery
            hir::db::InternStaticQuery
            hir::db::InternTraitQuery
            hir::db::InternTypeAliasQuery
            hir::db::InternImplQuery

            // HirDatabase
            hir::db::InferQueryQuery
            hir::db::TyQuery
            hir::db::ValueTyQuery
            hir::db::ImplSelfTyQuery
            hir::db::ImplTraitQuery
            hir::db::FieldTypesQuery
            hir::db::CallableItemSignatureQuery
            hir::db::GenericPredicatesForParamQuery
            hir::db::GenericPredicatesQuery
            hir::db::GenericDefaultsQuery
            hir::db::ImplsInCrateQuery
            hir::db::InternTypeCtorQuery
            hir::db::InternTypeParamIdQuery
            hir::db::InternChalkImplQuery
            hir::db::InternAssocTyValueQuery
            hir::db::AssociatedTyDataQuery
            hir::db::TraitDatumQuery
            hir::db::StructDatumQuery
            hir::db::ImplDatumQuery
            hir::db::AssociatedTyValueQuery
            hir::db::TraitSolveQuery
            hir::db::ReturnTypeImplTraitsQuery

            // SymbolsDatabase
            crate::symbol_index::FileSymbolsQuery

            // LineIndexDatabase
            crate::LineIndexQuery
        ];
        acc.sort_by_key(|it| std::cmp::Reverse(it.1));
        acc
    }
}

fn durability(source_root: &SourceRoot) -> Durability {
    if source_root.is_library {
        Durability::HIGH
    } else {
        Durability::LOW
    }
}
