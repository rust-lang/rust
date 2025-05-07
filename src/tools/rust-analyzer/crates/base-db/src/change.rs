//! Defines a unit of change that can applied to the database to get the next
//! state. Changes are transactional.

use std::fmt;

use salsa::Durability;
use triomphe::Arc;
use vfs::FileId;

use crate::{CrateGraphBuilder, CratesIdMap, RootQueryDb, SourceRoot, SourceRootId};

/// Encapsulate a bunch of raw `.set` calls on the database.
#[derive(Default)]
pub struct FileChange {
    pub roots: Option<Vec<SourceRoot>>,
    pub files_changed: Vec<(FileId, Option<String>)>,
    pub crate_graph: Option<CrateGraphBuilder>,
}

impl fmt::Debug for FileChange {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = fmt.debug_struct("Change");
        if let Some(roots) = &self.roots {
            d.field("roots", roots);
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

impl FileChange {
    pub fn set_roots(&mut self, roots: Vec<SourceRoot>) {
        self.roots = Some(roots);
    }

    pub fn change_file(&mut self, file_id: FileId, new_text: Option<String>) {
        self.files_changed.push((file_id, new_text))
    }

    pub fn set_crate_graph(&mut self, graph: CrateGraphBuilder) {
        self.crate_graph = Some(graph);
    }

    pub fn apply(self, db: &mut dyn RootQueryDb) -> Option<CratesIdMap> {
        let _p = tracing::info_span!("FileChange::apply").entered();
        if let Some(roots) = self.roots {
            for (idx, root) in roots.into_iter().enumerate() {
                let root_id = SourceRootId(idx as u32);
                let durability = source_root_durability(&root);
                for file_id in root.iter() {
                    db.set_file_source_root_with_durability(file_id, root_id, durability);
                }

                db.set_source_root_with_durability(root_id, Arc::new(root), durability);
            }
        }

        for (file_id, text) in self.files_changed {
            let source_root_id = db.file_source_root(file_id);
            let source_root = db.source_root(source_root_id.source_root_id(db));

            let durability = file_text_durability(&source_root.source_root(db));
            // XXX: can't actually remove the file, just reset the text
            let text = text.unwrap_or_default();
            db.set_file_text_with_durability(file_id, &text, durability)
        }

        if let Some(crate_graph) = self.crate_graph {
            return Some(crate_graph.set_in_db(db));
        }
        None
    }
}

fn source_root_durability(source_root: &SourceRoot) -> Durability {
    if source_root.is_library { Durability::MEDIUM } else { Durability::LOW }
}

fn file_text_durability(source_root: &SourceRoot) -> Durability {
    if source_root.is_library { Durability::HIGH } else { Durability::LOW }
}
