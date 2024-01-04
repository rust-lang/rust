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
pub struct ProcMacroId(u32);

impl ProcMacroId {
    pub fn new(u32: u32) -> Self {
        ProcMacroId(u32)
    }
}

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
    pub disabled: bool,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct CustomProcMacroExpander {
    proc_macro_id: ProcMacroId,
}

impl CustomProcMacroExpander {
    const DUMMY_ID: u32 = !0;
    const DISABLED_ID: u32 = !1;

    pub fn new(proc_macro_id: ProcMacroId) -> Self {
        assert_ne!(proc_macro_id.0, Self::DUMMY_ID);
        assert_ne!(proc_macro_id.0, Self::DISABLED_ID);
        Self { proc_macro_id }
    }

    /// A dummy expander that always errors. This is used for proc-macros that are missing, usually
    /// due to them not being built yet.
    pub const fn dummy() -> Self {
        Self { proc_macro_id: ProcMacroId(Self::DUMMY_ID) }
    }

    /// The macro was not yet resolved.
    pub const fn is_dummy(&self) -> bool {
        self.proc_macro_id.0 == Self::DUMMY_ID
    }

    /// A dummy expander that always errors. This expander is used for macros that have been disabled.
    pub const fn disabled() -> Self {
        Self { proc_macro_id: ProcMacroId(Self::DISABLED_ID) }
    }

    /// The macro is explicitly disabled and cannot be expanded.
    pub const fn is_disabled(&self) -> bool {
        self.proc_macro_id.0 == Self::DISABLED_ID
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
            ProcMacroId(Self::DUMMY_ID) => ExpandResult::new(
                tt::Subtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                ExpandError::UnresolvedProcMacro(def_crate),
            ),
            ProcMacroId(Self::DISABLED_ID) => ExpandResult::new(
                tt::Subtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                ExpandError::MacroDisabled,
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
                            ExpandError::other("Internal error: proc-macro index out of bounds"),
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
