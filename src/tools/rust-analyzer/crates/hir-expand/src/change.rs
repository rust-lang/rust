//! Defines a unit of change that can applied to the database to get the next
//! state. Changes are transactional.
use base_db::{
    salsa::Durability, CrateGraph, CrateId, FileChange, SourceDatabaseExt, SourceRoot,
    TargetLayoutLoadResult, Version,
};
use la_arena::RawIdx;
use span::FileId;
use triomphe::Arc;

use crate::{db::ExpandDatabase, proc_macro::ProcMacros};

#[derive(Debug, Default)]
pub struct Change {
    pub source_change: FileChange,
    pub proc_macros: Option<ProcMacros>,
    pub toolchains: Option<Vec<Option<Version>>>,
    pub target_data_layouts: Option<Vec<TargetLayoutLoadResult>>,
}

impl Change {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(self, db: &mut (impl ExpandDatabase + SourceDatabaseExt)) {
        self.source_change.apply(db);
        if let Some(proc_macros) = self.proc_macros {
            db.set_proc_macros_with_durability(Arc::new(proc_macros), Durability::HIGH);
        }
        if let Some(target_data_layouts) = self.target_data_layouts {
            for (id, val) in target_data_layouts.into_iter().enumerate() {
                db.set_data_layout_with_durability(
                    CrateId::from_raw(RawIdx::from(id as u32)),
                    val,
                    Durability::HIGH,
                );
            }
        }
        if let Some(toolchains) = self.toolchains {
            for (id, val) in toolchains.into_iter().enumerate() {
                db.set_toolchain_with_durability(
                    CrateId::from_raw(RawIdx::from(id as u32)),
                    val,
                    Durability::HIGH,
                );
            }
        }
    }

    pub fn change_file(&mut self, file_id: FileId, new_text: Option<Arc<str>>) {
        self.source_change.change_file(file_id, new_text)
    }

    pub fn set_crate_graph(&mut self, graph: CrateGraph) {
        self.source_change.set_crate_graph(graph)
    }

    pub fn set_proc_macros(&mut self, proc_macros: ProcMacros) {
        self.proc_macros = Some(proc_macros);
    }

    pub fn set_toolchains(&mut self, toolchains: Vec<Option<Version>>) {
        self.toolchains = Some(toolchains);
    }

    pub fn set_target_data_layouts(&mut self, target_data_layouts: Vec<TargetLayoutLoadResult>) {
        self.target_data_layouts = Some(target_data_layouts);
    }

    pub fn set_roots(&mut self, roots: Vec<SourceRoot>) {
        self.source_change.set_roots(roots)
    }
}
