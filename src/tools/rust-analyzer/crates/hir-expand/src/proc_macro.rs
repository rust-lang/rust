//! Proc Macro Expander stuff

use core::fmt;
use std::{panic::RefUnwindSafe, sync};

use base_db::{CrateId, Env};
use rustc_hash::FxHashMap;
use span::Span;
use stdx::never;
use syntax::SmolStr;

use crate::{db::ExpandDatabase, tt, ExpandError, ExpandResult};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ProcMacroId(pub u32);

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum ProcMacroKind {
    CustomDerive,
    FuncLike,
    Attr,
}

pub trait ProcMacroExpander: fmt::Debug + Send + Sync + RefUnwindSafe {
    fn expand(
        &self,
        subtree: &tt::Subtree,
        attrs: Option<&tt::Subtree>,
        env: &Env,
        def_site: Span,
        call_site: Span,
        mixed_site: Span,
    ) -> Result<tt::Subtree, ProcMacroExpansionError>;
}

#[derive(Debug)]
pub enum ProcMacroExpansionError {
    Panic(String),
    /// Things like "proc macro server was killed by OOM".
    System(String),
}

pub type ProcMacroLoadResult = Result<Vec<ProcMacro>, String>;

pub type ProcMacros = FxHashMap<CrateId, ProcMacroLoadResult>;

#[derive(Debug, Clone)]
pub struct ProcMacro {
    pub name: SmolStr,
    pub kind: ProcMacroKind,
    pub expander: sync::Arc<dyn ProcMacroExpander>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct CustomProcMacroExpander {
    proc_macro_id: ProcMacroId,
}

const DUMMY_ID: u32 = !0;

impl CustomProcMacroExpander {
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
        def_site: Span,
        call_site: Span,
        mixed_site: Span,
    ) -> ExpandResult<tt::Subtree> {
        match self.proc_macro_id {
            ProcMacroId(DUMMY_ID) => ExpandResult::new(
                tt::Subtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                ExpandError::UnresolvedProcMacro(def_crate),
            ),
            ProcMacroId(id) => {
                let proc_macros = db.proc_macros();
                let proc_macros = match proc_macros.get(&def_crate) {
                    Some(Ok(proc_macros)) => proc_macros,
                    Some(Err(_)) | None => {
                        never!("Non-dummy expander even though there are no proc macros");
                        return ExpandResult::new(
                            tt::Subtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
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
                            tt::Subtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                            ExpandError::other("Internal error"),
                        );
                    }
                };

                let krate_graph = db.crate_graph();
                // Proc macros have access to the environment variables of the invoking crate.
                let env = &krate_graph[calling_crate].env;
                match proc_macro.expander.expand(tt, attr_arg, env, def_site, call_site, mixed_site)
                {
                    Ok(t) => ExpandResult::ok(t),
                    Err(err) => match err {
                        // Don't discard the item in case something unexpected happened while expanding attributes
                        ProcMacroExpansionError::System(text)
                            if proc_macro.kind == ProcMacroKind::Attr =>
                        {
                            ExpandResult { value: tt.clone(), err: Some(ExpandError::other(text)) }
                        }
                        ProcMacroExpansionError::System(text)
                        | ProcMacroExpansionError::Panic(text) => ExpandResult::new(
                            tt::Subtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                            ExpandError::ProcMacroPanic(Box::new(text.into_boxed_str())),
                        ),
                    },
                }
            }
        }
    }
}
