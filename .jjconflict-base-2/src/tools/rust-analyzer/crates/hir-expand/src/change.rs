//! Defines a unit of change that can applied to the database to get the next
//! state. Changes are transactional.
use base_db::{CrateGraphBuilder, FileChange, SourceDatabase, SourceRoot};
use span::FileId;

use crate::proc_macro::ProcMacrosBuilder;

#[derive(Debug, Default)]
pub struct ChangeWithProcMacros {
    pub source_change: FileChange,
    pub proc_macros: Option<ProcMacrosBuilder>,
}

impl ChangeWithProcMacros {
    pub fn apply(self, db: &mut impl SourceDatabase) {
        let crates_id_map = self.source_change.apply(db);
        if let Some(proc_macros) = self.proc_macros {
            proc_macros.build_in(
                db,
                crates_id_map
                    .as_ref()
                    .expect("cannot set proc macros without setting the crate graph too"),
            );
        }
    }

    pub fn change_file(&mut self, file_id: FileId, new_text: Option<String>) {
        self.source_change.change_file(file_id, new_text)
    }

    pub fn set_crate_graph(&mut self, graph: CrateGraphBuilder) {
        self.source_change.set_crate_graph(graph);
    }

    pub fn set_proc_macros(&mut self, proc_macros: ProcMacrosBuilder) {
        self.proc_macros = Some(proc_macros);
    }

    pub fn set_roots(&mut self, roots: Vec<SourceRoot>) {
        self.source_change.set_roots(roots)
    }
}
