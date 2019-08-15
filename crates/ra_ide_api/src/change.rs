use std::{fmt, sync::Arc, time};

use ra_db::{
    salsa::{Database, Durability, SweepStrategy},
    CrateGraph, FileId, SourceDatabase, SourceRoot, SourceRootId,
};
use ra_prof::{memory_usage, profile, Bytes};
use ra_syntax::SourceFile;
use rayon::prelude::*;
use relative_path::RelativePathBuf;
use rustc_hash::FxHashMap;

use crate::{
    db::RootDatabase,
    status::syntax_tree_stats,
    symbol_index::{SymbolIndex, SymbolsDatabase},
};

#[derive(Default)]
pub struct AnalysisChange {
    new_roots: Vec<(SourceRootId, bool)>,
    roots_changed: FxHashMap<SourceRootId, RootChange>,
    files_changed: Vec<(FileId, Arc<String>)>,
    libraries_added: Vec<LibraryData>,
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
        if !self.libraries_added.is_empty() {
            d.field("libraries_added", &self.libraries_added.len());
        }
        if !self.crate_graph.is_some() {
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

    pub fn add_library(&mut self, data: LibraryData) {
        self.libraries_added.push(data)
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

pub struct LibraryData {
    root_id: SourceRootId,
    root_change: RootChange,
    symbol_index: SymbolIndex,
}

impl fmt::Debug for LibraryData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LibraryData")
            .field("root_id", &self.root_id)
            .field("root_change", &self.root_change)
            .field("n_symbols", &self.symbol_index.len())
            .finish()
    }
}

impl LibraryData {
    pub fn prepare(
        root_id: SourceRootId,
        files: Vec<(FileId, RelativePathBuf, Arc<String>)>,
    ) -> LibraryData {
        let symbol_index = SymbolIndex::for_files(files.par_iter().map(|(file_id, _, text)| {
            let parse = SourceFile::parse(text);
            (*file_id, parse)
        }));
        let mut root_change = RootChange::default();
        root_change.added = files
            .into_iter()
            .map(|(file_id, path, text)| AddFile { file_id, path, text })
            .collect();
        LibraryData { root_id, root_change, symbol_index }
    }
}

const GC_COOLDOWN: time::Duration = time::Duration::from_millis(100);

impl RootDatabase {
    pub(crate) fn apply_change(&mut self, change: AnalysisChange) {
        let _p = profile("RootDatabase::apply_change");
        log::info!("apply_change {:?}", change);
        {
            let _p = profile("RootDatabase::apply_change/cancellation");
            self.salsa_runtime().synthetic_write(Durability::LOW);
        }
        if !change.new_roots.is_empty() {
            let mut local_roots = Vec::clone(&self.local_roots());
            for (root_id, is_local) in change.new_roots {
                let root = if is_local { SourceRoot::new() } else { SourceRoot::new_library() };
                let durability = durability(&root);
                self.set_source_root_with_durability(root_id, Arc::new(root), durability);
                if is_local {
                    local_roots.push(root_id);
                }
            }
            self.set_local_roots_with_durability(Arc::new(local_roots), Durability::HIGH);
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
        if !change.libraries_added.is_empty() {
            let mut libraries = Vec::clone(&self.library_roots());
            for library in change.libraries_added {
                libraries.push(library.root_id);
                self.set_source_root_with_durability(
                    library.root_id,
                    Default::default(),
                    Durability::HIGH,
                );
                self.set_library_symbols_with_durability(
                    library.root_id,
                    Arc::new(library.symbol_index),
                    Durability::HIGH,
                );
                self.apply_root_change(library.root_id, library.root_change);
            }
            self.set_library_roots_with_durability(Arc::new(libraries), Durability::HIGH);
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
            source_root.files.insert(add_file.path, add_file.file_id);
        }
        for remove_file in root_change.removed {
            self.set_file_text_with_durability(remove_file.file_id, Default::default(), durability);
            source_root.files.remove(&remove_file.path);
        }
        self.set_source_root_with_durability(root_id, Arc::new(source_root), durability);
    }

    pub(crate) fn maybe_collect_garbage(&mut self) {
        if self.last_gc_check.elapsed() > GC_COOLDOWN {
            self.last_gc_check = time::Instant::now();
            let retained_trees = syntax_tree_stats(self).retained;
            if retained_trees > 100 {
                log::info!("automatic garbadge collection, {} retained trees", retained_trees);
                self.collect_garbage();
            }
        }
    }

    pub(crate) fn collect_garbage(&mut self) {
        let _p = profile("RootDatabase::collect_garbage");
        self.last_gc = time::Instant::now();

        let sweep = SweepStrategy::default().discard_values().sweep_all_revisions();

        self.query(ra_db::ParseQuery).sweep(sweep);
        self.query(hir::db::ParseMacroQuery).sweep(sweep);

        // Macros do take significant space, but less then the syntax trees
        // self.query(hir::db::MacroDefQuery).sweep(sweep);
        // self.query(hir::db::MacroArgQuery).sweep(sweep);
        // self.query(hir::db::MacroExpandQuery).sweep(sweep);

        self.query(hir::db::AstIdMapQuery).sweep(sweep);

        self.query(hir::db::RawItemsWithSourceMapQuery).sweep(sweep);
        self.query(hir::db::ImplsInModuleWithSourceMapQuery).sweep(sweep);
        self.query(hir::db::BodyWithSourceMapQuery).sweep(sweep);

        self.query(hir::db::ExprScopesQuery).sweep(sweep);
        self.query(hir::db::InferQuery).sweep(sweep);
        self.query(hir::db::BodyHirQuery).sweep(sweep);
    }

    pub(crate) fn per_query_memory_usage(&mut self) -> Vec<(String, Bytes)> {
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
            ra_db::ParseQuery
            ra_db::SourceRootCratesQuery
            hir::db::AstIdMapQuery
            hir::db::ParseMacroQuery
            hir::db::MacroDefQuery
            hir::db::MacroArgQuery
            hir::db::MacroExpandQuery
            hir::db::StructDataQuery
            hir::db::EnumDataQuery
            hir::db::TraitDataQuery
            hir::db::TraitItemsIndexQuery
            hir::db::RawItemsWithSourceMapQuery
            hir::db::RawItemsQuery
            hir::db::CrateDefMapQuery
            hir::db::ImplsInModuleWithSourceMapQuery
            hir::db::ImplsInModuleQuery
            hir::db::GenericParamsQuery
            hir::db::FnDataQuery
            hir::db::TypeAliasDataQuery
            hir::db::ConstDataQuery
            hir::db::StaticDataQuery
            hir::db::ModuleLangItemsQuery
            hir::db::LangItemsQuery
            hir::db::LangItemQuery
            hir::db::DocumentationQuery
            hir::db::ExprScopesQuery
            hir::db::InferQuery
            hir::db::TypeForDefQuery
            hir::db::TypeForFieldQuery
            hir::db::CallableItemSignatureQuery
            hir::db::GenericPredicatesQuery
            hir::db::GenericDefaultsQuery
            hir::db::BodyWithSourceMapQuery
            hir::db::BodyHirQuery
            hir::db::ImplsInCrateQuery
            hir::db::ImplsForTraitQuery
            hir::db::AssociatedTyDataQuery
            hir::db::TraitDatumQuery
            hir::db::StructDatumQuery
            hir::db::ImplDatumQuery
            hir::db::TraitSolveQuery
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
