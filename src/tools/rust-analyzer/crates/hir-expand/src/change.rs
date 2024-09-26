//! Defines a unit of change that can applied to the database to get the next
//! state. Changes are transactional.
use base_db::{
    salsa::Durability, CrateGraph, CrateId, CrateWorkspaceData, FileChange, SourceRoot,
    SourceRootDatabase,
};
use rustc_hash::FxHashMap;
use span::FileId;
use triomphe::Arc;

use crate::{db::ExpandDatabase, proc_macro::ProcMacros};

#[derive(Debug, Default)]
pub struct ChangeWithProcMacros {
    pub source_change: FileChange,
    pub proc_macros: Option<ProcMacros>,
}

impl ChangeWithProcMacros {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(self, db: &mut (impl ExpandDatabase + SourceRootDatabase)) {
        self.source_change.apply(db);
        if let Some(proc_macros) = self.proc_macros {
            db.set_proc_macros_with_durability(Arc::new(proc_macros), Durability::HIGH);
        }
    }

    pub fn change_file(&mut self, file_id: FileId, new_text: Option<String>) {
        self.source_change.change_file(file_id, new_text)
    }

    pub fn set_crate_graph(
        &mut self,
        graph: CrateGraph,
        ws_data: FxHashMap<CrateId, Arc<CrateWorkspaceData>>,
    ) {
        self.source_change.set_crate_graph(graph);
        self.source_change.set_ws_data(ws_data);
    }

    pub fn set_proc_macros(&mut self, proc_macros: ProcMacros) {
        self.proc_macros = Some(proc_macros);
    }

    pub fn set_roots(&mut self, roots: Vec<SourceRoot>) {
        self.source_change.set_roots(roots)
    }
}
