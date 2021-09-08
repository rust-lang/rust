//! This module provides `StaticIndex` which is used for powering
//! read-only code browsers and emitting LSIF

use hir::{db::HirDatabase, Crate, Module};
use ide_db::base_db::{FileId, SourceDatabaseExt};
use ide_db::RootDatabase;
use rustc_hash::FxHashSet;

use crate::{Analysis, Cancellable, Fold};

/// A static representation of fully analyzed source code.
///
/// The intended use-case is powering read-only code browsers and emitting LSIF
pub struct StaticIndex {
    pub files: Vec<StaticIndexedFile>,
}

pub struct StaticIndexedFile {
    pub file_id: FileId,
    pub folds: Vec<Fold>,
}

fn all_modules(db: &dyn HirDatabase) -> Vec<Module> {
    let mut worklist: Vec<_> =
        Crate::all(db).into_iter().map(|krate| krate.root_module(db)).collect();
    let mut modules = Vec::new();

    while let Some(module) = worklist.pop() {
        modules.push(module);
        worklist.extend(module.children(db));
    }

    modules
}

impl StaticIndex {
    pub fn compute(db: &RootDatabase, analysis: &Analysis) -> Cancellable<StaticIndex> {
        let work = all_modules(db).into_iter().filter(|module| {
            let file_id = module.definition_source(db).file_id.original_file(db);
            let source_root = db.file_source_root(file_id);
            let source_root = db.source_root(source_root);
            !source_root.is_library
        });

        let mut visited_files = FxHashSet::default();
        let mut result_files = Vec::<StaticIndexedFile>::new();
        for module in work {
            let file_id = module.definition_source(db).file_id.original_file(db);
            if !visited_files.contains(&file_id) {
                //let path = vfs.file_path(file_id);
                //let path = path.as_path().unwrap();
                //let doc_id = lsif.add(Element::Vertex(Vertex::Document(Document {
                //    language_id: Language::Rust,
                //    uri: lsp_types::Url::from_file_path(path).unwrap(),
                //})));
                let folds = analysis.folding_ranges(file_id)?;
                result_files.push(StaticIndexedFile { file_id, folds });
                visited_files.insert(file_id);
            }
        }
        Ok(StaticIndex { files: result_files })
    }
}
