//! Proc Macro Expander stub

use crate::{db::AstDatabase, LazyMacroId};
use ra_db::{CrateId, ProcMacroId};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ProcMacroExpander {
    krate: CrateId,
    proc_macro_id: ProcMacroId,
}

macro_rules! err {
    ($fmt:literal, $($tt:tt),*) => {
        mbe::ExpandError::ProcMacroError(tt::ExpansionError::Unknown(format!($fmt, $($tt),*)))
    };
    ($fmt:literal) => {
        mbe::ExpandError::ProcMacroError(tt::ExpansionError::Unknown($fmt.to_string()))
    }
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
            .get(self.proc_macro_id.0 as usize)
            .clone()
            .ok_or_else(|| err!("No derive macro found."))?;

        let tt = remove_derive_atr(tt, &proc_macro.name)
            .ok_or_else(|| err!("Fail to remove derive for custom derive"))?;

        proc_macro.expander.expand(&tt, None).map_err(mbe::ExpandError::from)
    }
}

fn remove_derive_atr(tt: &tt::Subtree, _name: &str) -> Option<tt::Subtree> {
    // FIXME: proper handle the remove derive
    // We assume the first 2 tokens are #[derive(name)]
    if tt.token_trees.len() > 2 {
        let mut tt = tt.clone();
        tt.token_trees.remove(0);
        tt.token_trees.remove(0);
        return Some(tt);
    }

    None
}
