//! Proc Macro Expander stuff

use core::fmt;
use std::{panic::RefUnwindSafe, sync};

use base_db::{CrateId, Env};
use intern::Symbol;
use rustc_hash::FxHashMap;
use span::Span;

use crate::{db::ExpandDatabase, tt, ExpandError, ExpandErrorKind, ExpandResult};

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum ProcMacroKind {
    CustomDerive,
    Bang,
    Attr,
}

/// A proc-macro expander implementation.
pub trait ProcMacroExpander: fmt::Debug + Send + Sync + RefUnwindSafe {
    /// Run the expander with the given input subtree, optional attribute input subtree (for
    /// [`ProcMacroKind::Attr`]), environment variables, and span information.
    fn expand(
        &self,
        subtree: &tt::Subtree,
        attrs: Option<&tt::Subtree>,
        env: &Env,
        def_site: Span,
        call_site: Span,
        mixed_site: Span,
        current_dir: Option<String>,
    ) -> Result<tt::Subtree, ProcMacroExpansionError>;
}

#[derive(Debug)]
pub enum ProcMacroExpansionError {
    /// The proc-macro panicked.
    Panic(String),
    /// The server itself errored out.
    System(String),
}

pub type ProcMacroLoadResult = Result<Vec<ProcMacro>, (String, bool)>;
type StoredProcMacroLoadResult = Result<Box<[ProcMacro]>, (Box<str>, bool)>;

#[derive(Default, Debug)]
pub struct ProcMacrosBuilder(FxHashMap<CrateId, StoredProcMacroLoadResult>);
impl ProcMacrosBuilder {
    pub fn insert(&mut self, proc_macros_crate: CrateId, proc_macro: ProcMacroLoadResult) {
        self.0.insert(
            proc_macros_crate,
            match proc_macro {
                Ok(it) => Ok(it.into_boxed_slice()),
                Err((e, hard_err)) => Err((e.into_boxed_str(), hard_err)),
            },
        );
    }
    pub fn build(mut self) -> ProcMacros {
        self.0.shrink_to_fit();
        ProcMacros(self.0)
    }
}

#[derive(Default, Debug)]
pub struct ProcMacros(FxHashMap<CrateId, StoredProcMacroLoadResult>);

impl FromIterator<(CrateId, ProcMacroLoadResult)> for ProcMacros {
    fn from_iter<T: IntoIterator<Item = (CrateId, ProcMacroLoadResult)>>(iter: T) -> Self {
        let mut builder = ProcMacrosBuilder::default();
        for (k, v) in iter {
            builder.insert(k, v);
        }
        builder.build()
    }
}

impl ProcMacros {
    fn get(&self, krate: CrateId, idx: u32, err_span: Span) -> Result<&ProcMacro, ExpandError> {
        let proc_macros = match self.0.get(&krate) {
            Some(Ok(proc_macros)) => proc_macros,
            Some(Err(_)) | None => {
                return Err(ExpandError::other(
                    err_span,
                    "internal error: no proc macros for crate",
                ));
            }
        };
        proc_macros.get(idx as usize).ok_or_else(|| {
                ExpandError::other(err_span,
                    format!(
                        "internal error: proc-macro index out of bounds: the length is {} but the index is {}",
                        proc_macros.len(),
                        idx
                    )
                )
            }
        )
    }

    pub fn get_error_for_crate(&self, krate: CrateId) -> Option<(&str, bool)> {
        self.0.get(&krate).and_then(|it| it.as_ref().err()).map(|(e, hard_err)| (&**e, *hard_err))
    }

    /// Fetch the [`CustomProcMacroExpander`]s and their corresponding names for the given crate.
    pub fn for_crate(
        &self,
        krate: CrateId,
        def_site_ctx: span::SyntaxContextId,
    ) -> Option<Box<[(crate::name::Name, CustomProcMacroExpander, bool)]>> {
        match self.0.get(&krate) {
            Some(Ok(proc_macros)) => Some({
                proc_macros
                    .iter()
                    .enumerate()
                    .map(|(idx, it)| {
                        let name = crate::name::Name::new_symbol(it.name.clone(), def_site_ctx);
                        (name, CustomProcMacroExpander::new(idx as u32), it.disabled)
                    })
                    .collect()
            }),
            _ => None,
        }
    }
}

/// A loaded proc-macro.
#[derive(Debug, Clone)]
pub struct ProcMacro {
    /// The name of the proc macro.
    pub name: Symbol,
    pub kind: ProcMacroKind,
    /// The expander handle for this proc macro.
    pub expander: sync::Arc<dyn ProcMacroExpander>,
    /// Whether this proc-macro is disabled for early name resolution. Notably, the
    /// [`Self::expander`] is still usable.
    pub disabled: bool,
}

/// A custom proc-macro expander handle. This handle together with its crate resolves to a [`ProcMacro`]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct CustomProcMacroExpander {
    proc_macro_id: u32,
}

impl CustomProcMacroExpander {
    const MISSING_EXPANDER: u32 = !0;
    const DISABLED_ID: u32 = !1;
    const PROC_MACRO_ATTR_DISABLED: u32 = !2;

    pub fn new(proc_macro_id: u32) -> Self {
        assert_ne!(proc_macro_id, Self::MISSING_EXPANDER);
        assert_ne!(proc_macro_id, Self::DISABLED_ID);
        assert_ne!(proc_macro_id, Self::PROC_MACRO_ATTR_DISABLED);
        Self { proc_macro_id }
    }

    /// An expander that always errors due to the actual proc-macro expander missing.
    pub const fn missing_expander() -> Self {
        Self { proc_macro_id: Self::MISSING_EXPANDER }
    }

    /// A dummy expander that always errors. This expander is used for macros that have been disabled.
    pub const fn disabled() -> Self {
        Self { proc_macro_id: Self::DISABLED_ID }
    }

    /// A dummy expander that always errors. This expander is used for attribute macros when
    /// proc-macro attribute expansion is disabled.
    pub const fn disabled_proc_attr() -> Self {
        Self { proc_macro_id: Self::PROC_MACRO_ATTR_DISABLED }
    }

    /// The macro-expander is missing or has yet to be build.
    pub const fn is_missing(&self) -> bool {
        self.proc_macro_id == Self::MISSING_EXPANDER
    }

    /// The macro is explicitly disabled and cannot be expanded.
    pub const fn is_disabled(&self) -> bool {
        self.proc_macro_id == Self::DISABLED_ID
    }

    /// The macro is explicitly disabled due to proc-macro attribute expansion being disabled.
    pub const fn is_disabled_proc_attr(&self) -> bool {
        self.proc_macro_id == Self::PROC_MACRO_ATTR_DISABLED
    }

    /// The macro is explicitly disabled due to proc-macro attribute expansion being disabled.
    pub fn as_expand_error(&self, def_crate: CrateId) -> Option<ExpandErrorKind> {
        match self.proc_macro_id {
            Self::PROC_MACRO_ATTR_DISABLED => Some(ExpandErrorKind::ProcMacroAttrExpansionDisabled),
            Self::DISABLED_ID => Some(ExpandErrorKind::MacroDisabled),
            Self::MISSING_EXPANDER => Some(ExpandErrorKind::MissingProcMacroExpander(def_crate)),
            _ => None,
        }
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
            Self::PROC_MACRO_ATTR_DISABLED => ExpandResult::new(
                tt::Subtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                ExpandError::new(call_site, ExpandErrorKind::ProcMacroAttrExpansionDisabled),
            ),
            Self::MISSING_EXPANDER => ExpandResult::new(
                tt::Subtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                ExpandError::new(call_site, ExpandErrorKind::MissingProcMacroExpander(def_crate)),
            ),
            Self::DISABLED_ID => ExpandResult::new(
                tt::Subtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                ExpandError::new(call_site, ExpandErrorKind::MacroDisabled),
            ),
            id => {
                let proc_macros = db.proc_macros();
                let proc_macro = match proc_macros.get(def_crate, id, call_site) {
                    Ok(proc_macro) => proc_macro,
                    Err(e) => {
                        return ExpandResult::new(
                            tt::Subtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                            e,
                        )
                    }
                };

                let krate_graph = db.crate_graph();
                // Proc macros have access to the environment variables of the invoking crate.
                let env = &krate_graph[calling_crate].env;
                match proc_macro.expander.expand(
                    tt,
                    attr_arg,
                    env,
                    def_site,
                    call_site,
                    mixed_site,
                    db.crate_workspace_data()[&calling_crate]
                        .proc_macro_cwd
                        .as_ref()
                        .map(ToString::to_string),
                ) {
                    Ok(t) => ExpandResult::ok(t),
                    Err(err) => match err {
                        // Don't discard the item in case something unexpected happened while expanding attributes
                        ProcMacroExpansionError::System(text)
                            if proc_macro.kind == ProcMacroKind::Attr =>
                        {
                            ExpandResult {
                                value: tt.clone(),
                                err: Some(ExpandError::other(call_site, text)),
                            }
                        }
                        ProcMacroExpansionError::System(text)
                        | ProcMacroExpansionError::Panic(text) => ExpandResult::new(
                            tt::Subtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                            ExpandError::new(
                                call_site,
                                ExpandErrorKind::ProcMacroPanic(text.into_boxed_str()),
                            ),
                        ),
                    },
                }
            }
        }
    }
}
