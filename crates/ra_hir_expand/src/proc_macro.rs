//! Proc Macro Expander stub

use crate::{db::AstDatabase, LazyMacroId};
use ra_db::{CrateId, ProcMacroId};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ProcMacroExpander {
    krate: CrateId,
    proc_macro_id: ProcMacroId,
}

impl ProcMacroExpander {
    pub fn new(krate: CrateId, proc_macro_id: ProcMacroId) -> ProcMacroExpander {
        ProcMacroExpander { krate, proc_macro_id }
    }

    pub fn expand(
        &self,
        db: &dyn AstDatabase,
        _id: LazyMacroId,
        tt: &tt::Subtree,
    ) -> Result<tt::Subtree, mbe::ExpandError> {
        let krate_graph = db.crate_graph();
        let proc_macro = krate_graph[self.krate]
            .proc_macro
            .get(self.proc_macro_id.0)
            .clone()
            .ok_or_else(|| mbe::ExpandError::ConversionError)?;
        proc_macro.custom_derive(tt)
    }
}
