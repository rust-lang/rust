//! Proc Macro Expander stub

use base_db::{CrateId, ProcMacroExpansionError, ProcMacroId, ProcMacroKind};
use stdx::never;

use crate::{db::ExpandDatabase, tt, ExpandError, ExpandResult};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ProcMacroExpander {
    proc_macro_id: ProcMacroId,
}

const DUMMY_ID: u32 = !0;

impl ProcMacroExpander {
    pub fn new(proc_macro_id: ProcMacroId) -> Self {
        assert_ne!(proc_macro_id.0, DUMMY_ID);
        Self { proc_macro_id }
    }

    pub fn dummy() -> Self {
        Self { proc_macro_id: ProcMacroId(DUMMY_ID) }
    }

    pub fn is_dummy(&self) -> bool {
        self.proc_macro_id.0 == DUMMY_ID
    }

    pub fn expand(
        self,
        db: &dyn ExpandDatabase,
        def_crate: CrateId,
        calling_crate: CrateId,
        tt: &tt::Subtree,
        attr_arg: Option<&tt::Subtree>,
    ) -> ExpandResult<tt::Subtree> {
        match self.proc_macro_id {
            ProcMacroId(DUMMY_ID) => {
                ExpandResult::new(tt::Subtree::empty(), ExpandError::UnresolvedProcMacro(def_crate))
            }
            ProcMacroId(id) => {
                let proc_macros = db.proc_macros();
                let proc_macros = match proc_macros.get(&def_crate) {
                    Some(Ok(proc_macros)) => proc_macros,
                    Some(Err(_)) | None => {
                        never!("Non-dummy expander even though there are no proc macros");
                        return ExpandResult::new(
                            tt::Subtree::empty(),
                            ExpandError::other("Internal error"),
                        );
                    }
                };
                let proc_macro = match proc_macros.get(id as usize) {
                    Some(proc_macro) => proc_macro,
                    None => {
                        never!(
                            "Proc macro index out of bounds: the length is {} but the index is {}",
                            proc_macros.len(),
                            id
                        );
                        return ExpandResult::new(
                            tt::Subtree::empty(),
                            ExpandError::other("Internal error"),
                        );
                    }
                };

                let krate_graph = db.crate_graph();
                // Proc macros have access to the environment variables of the invoking crate.
                let env = &krate_graph[calling_crate].env;
                match proc_macro.expander.expand(tt, attr_arg, env) {
                    Ok(t) => ExpandResult::ok(t),
                    Err(err) => match err {
                        // Don't discard the item in case something unexpected happened while expanding attributes
                        ProcMacroExpansionError::System(text)
                            if proc_macro.kind == ProcMacroKind::Attr =>
                        {
                            ExpandResult { value: tt.clone(), err: Some(ExpandError::other(text)) }
                        }
                        ProcMacroExpansionError::System(text)
                        | ProcMacroExpansionError::Panic(text) => {
                            ExpandResult::new(tt::Subtree::empty(), ExpandError::other(text))
                        }
                    },
                }
            }
        }
    }
}
