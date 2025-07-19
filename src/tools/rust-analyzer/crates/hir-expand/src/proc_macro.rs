//! Proc Macro Expander stuff

use core::fmt;
use std::any::Any;
use std::{panic::RefUnwindSafe, sync};

use base_db::{Crate, CrateBuilderId, CratesIdMap, Env, ProcMacroLoadingError};
use intern::Symbol;
use rustc_hash::FxHashMap;
use span::Span;
use triomphe::Arc;

use crate::{ExpandError, ExpandErrorKind, ExpandResult, db::ExpandDatabase, tt};

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Debug, Hash)]
pub enum ProcMacroKind {
    CustomDerive,
    Bang,
    Attr,
}

/// A proc-macro expander implementation.
pub trait ProcMacroExpander: fmt::Debug + Send + Sync + RefUnwindSafe + Any {
    /// Run the expander with the given input subtree, optional attribute input subtree (for
    /// [`ProcMacroKind::Attr`]), environment variables, and span information.
    fn expand(
        &self,
        subtree: &tt::TopSubtree,
        attrs: Option<&tt::TopSubtree>,
        env: &Env,
        def_site: Span,
        call_site: Span,
        mixed_site: Span,
        current_dir: String,
    ) -> Result<tt::TopSubtree, ProcMacroExpansionError>;

    fn eq_dyn(&self, other: &dyn ProcMacroExpander) -> bool;
}

impl PartialEq for dyn ProcMacroExpander {
    fn eq(&self, other: &Self) -> bool {
        self.eq_dyn(other)
    }
}

impl Eq for dyn ProcMacroExpander {}

#[derive(Debug)]
pub enum ProcMacroExpansionError {
    /// The proc-macro panicked.
    Panic(String),
    /// The server itself errored out.
    System(String),
}

pub type ProcMacroLoadResult = Result<Vec<ProcMacro>, ProcMacroLoadingError>;
type StoredProcMacroLoadResult = Result<Box<[ProcMacro]>, ProcMacroLoadingError>;

#[derive(Default, Debug)]
pub struct ProcMacrosBuilder(FxHashMap<CrateBuilderId, Arc<CrateProcMacros>>);

impl ProcMacrosBuilder {
    pub fn insert(
        &mut self,
        proc_macros_crate: CrateBuilderId,
        mut proc_macro: ProcMacroLoadResult,
    ) {
        if let Ok(proc_macros) = &mut proc_macro {
            // Sort proc macros to improve incrementality when only their order has changed (ideally the build system
            // will not change their order, but just to be sure).
            proc_macros.sort_unstable_by(|proc_macro, proc_macro2| {
                (proc_macro.name.as_str(), proc_macro.kind)
                    .cmp(&(proc_macro2.name.as_str(), proc_macro2.kind))
            });
        }
        self.0.insert(
            proc_macros_crate,
            match proc_macro {
                Ok(it) => Arc::new(CrateProcMacros(Ok(it.into_boxed_slice()))),
                Err(e) => Arc::new(CrateProcMacros(Err(e))),
            },
        );
    }

    pub(crate) fn build(self, crates_id_map: &CratesIdMap) -> ProcMacros {
        let mut map = self
            .0
            .into_iter()
            .map(|(krate, proc_macro)| (crates_id_map[&krate], proc_macro))
            .collect::<FxHashMap<_, _>>();
        map.shrink_to_fit();
        ProcMacros(map)
    }
}

impl FromIterator<(CrateBuilderId, ProcMacroLoadResult)> for ProcMacrosBuilder {
    fn from_iter<T: IntoIterator<Item = (CrateBuilderId, ProcMacroLoadResult)>>(iter: T) -> Self {
        let mut builder = ProcMacrosBuilder::default();
        for (k, v) in iter {
            builder.insert(k, v);
        }
        builder
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct CrateProcMacros(StoredProcMacroLoadResult);

#[derive(Default, Debug)]
pub struct ProcMacros(FxHashMap<Crate, Arc<CrateProcMacros>>);
impl ProcMacros {
    fn get(&self, krate: Crate) -> Option<Arc<CrateProcMacros>> {
        self.0.get(&krate).cloned()
    }
}

impl CrateProcMacros {
    fn get(&self, idx: u32, err_span: Span) -> Result<&ProcMacro, ExpandError> {
        let proc_macros = match &self.0 {
            Ok(proc_macros) => proc_macros,
            Err(_) => {
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

    pub fn get_error(&self) -> Option<&ProcMacroLoadingError> {
        self.0.as_ref().err()
    }

    /// Fetch the [`CustomProcMacroExpander`]s and their corresponding names for the given crate.
    pub fn list(
        &self,
        def_site_ctx: span::SyntaxContext,
    ) -> Option<Box<[(crate::name::Name, CustomProcMacroExpander, bool)]>> {
        match &self.0 {
            Ok(proc_macros) => Some(
                proc_macros
                    .iter()
                    .enumerate()
                    .map(|(idx, it)| {
                        let name = crate::name::Name::new_symbol(it.name.clone(), def_site_ctx);
                        (name, CustomProcMacroExpander::new(idx as u32), it.disabled)
                    })
                    .collect(),
            ),
            _ => None,
        }
    }
}

/// A loaded proc-macro.
#[derive(Debug, Clone, Eq)]
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

// `#[derive(PartialEq)]` generates a strange "cannot move" error.
impl PartialEq for ProcMacro {
    fn eq(&self, other: &Self) -> bool {
        let Self { name, kind, expander, disabled } = self;
        let Self {
            name: other_name,
            kind: other_kind,
            expander: other_expander,
            disabled: other_disabled,
        } = other;
        name == other_name
            && kind == other_kind
            && expander == other_expander
            && disabled == other_disabled
    }
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

    pub fn as_expand_error(&self, def_crate: Crate) -> Option<ExpandErrorKind> {
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
        def_crate: Crate,
        calling_crate: Crate,
        tt: &tt::TopSubtree,
        attr_arg: Option<&tt::TopSubtree>,
        def_site: Span,
        call_site: Span,
        mixed_site: Span,
    ) -> ExpandResult<tt::TopSubtree> {
        match self.proc_macro_id {
            Self::PROC_MACRO_ATTR_DISABLED => ExpandResult::new(
                tt::TopSubtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                ExpandError::new(call_site, ExpandErrorKind::ProcMacroAttrExpansionDisabled),
            ),
            Self::MISSING_EXPANDER => ExpandResult::new(
                tt::TopSubtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                ExpandError::new(call_site, ExpandErrorKind::MissingProcMacroExpander(def_crate)),
            ),
            Self::DISABLED_ID => ExpandResult::new(
                tt::TopSubtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                ExpandError::new(call_site, ExpandErrorKind::MacroDisabled),
            ),
            id => {
                let proc_macros = match db.proc_macros_for_crate(def_crate) {
                    Some(it) => it,
                    None => {
                        return ExpandResult::new(
                            tt::TopSubtree::empty(tt::DelimSpan {
                                open: call_site,
                                close: call_site,
                            }),
                            ExpandError::other(
                                call_site,
                                "internal error: no proc macros for crate",
                            ),
                        );
                    }
                };
                let proc_macro = match proc_macros.get(id, call_site) {
                    Ok(proc_macro) => proc_macro,
                    Err(e) => {
                        return ExpandResult::new(
                            tt::TopSubtree::empty(tt::DelimSpan {
                                open: call_site,
                                close: call_site,
                            }),
                            e,
                        );
                    }
                };

                // Proc macros have access to the environment variables of the invoking crate.
                let env = calling_crate.env(db);
                // FIXME: Can we avoid the string allocation here?
                let current_dir = calling_crate.data(db).proc_macro_cwd.to_string();

                match proc_macro.expander.expand(
                    tt,
                    attr_arg,
                    env,
                    def_site,
                    call_site,
                    mixed_site,
                    current_dir,
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
                            tt::TopSubtree::empty(tt::DelimSpan {
                                open: call_site,
                                close: call_site,
                            }),
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

pub(crate) fn proc_macros_for_crate(
    db: &dyn ExpandDatabase,
    krate: Crate,
) -> Option<Arc<CrateProcMacros>> {
    db.proc_macros().get(krate)
}
