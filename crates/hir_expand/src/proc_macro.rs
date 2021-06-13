//! Proc Macro Expander stub

use crate::db::AstDatabase;
use base_db::{CrateId, ProcMacroId};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ProcMacroExpander {
    krate: CrateId,
    proc_macro_id: Option<ProcMacroId>,
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
    pub fn new(krate: CrateId, proc_macro_id: ProcMacroId) -> Self {
        Self { krate, proc_macro_id: Some(proc_macro_id) }
    }

    pub fn dummy(krate: CrateId) -> Self {
        // FIXME: Should store the name for better errors
        Self { krate, proc_macro_id: None }
    }

    pub fn is_dummy(&self) -> bool {
        self.proc_macro_id.is_none()
    }

    pub fn expand(
        self,
        db: &dyn AstDatabase,
        calling_crate: CrateId,
        tt: &tt::Subtree,
        attr_arg: Option<&tt::Subtree>,
    ) -> Result<tt::Subtree, mbe::ExpandError> {
        match self.proc_macro_id {
            Some(id) => {
                let krate_graph = db.crate_graph();
                let proc_macro = krate_graph[self.krate]
                    .proc_macro
                    .get(id.0 as usize)
                    .ok_or_else(|| err!("No derive macro found."))?;

                // Proc macros have access to the environment variables of the invoking crate.
                let env = &krate_graph[calling_crate].env;

                proc_macro.expander.expand(tt, attr_arg, env).map_err(mbe::ExpandError::from)
            }
            None => Err(mbe::ExpandError::UnresolvedProcMacro),
        }
    }
}
