pub mod bug;
pub mod call_kind;
pub mod common;
pub mod find_self_call;

use std::env::VarError;
use std::ffi::OsStr;

pub use call_kind::{CallDesugaringKind, CallKind, call_kind};
pub use find_self_call::find_self_call;

use crate::ty::TyCtxt;

#[derive(Default, Copy, Clone)]
pub struct Providers {
    pub queries: rustc_middle::query::Providers,
    pub extern_queries: rustc_middle::query::ExternProviders,
    pub hooks: rustc_middle::hooks::Providers,
}

/// Backwards compatibility hack to keep the diff small. This
/// gives direct access to the `queries` field's fields, which
/// are what almost everything wants access to.
impl std::ops::DerefMut for Providers {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.queries
    }
}

impl std::ops::Deref for Providers {
    type Target = rustc_middle::query::Providers;

    fn deref(&self) -> &Self::Target {
        &self.queries
    }
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn env_var(self, key: &'tcx OsStr) -> Result<&'tcx str, VarError> {
        if let Some(value) = self.env_var_os(key.as_ref()) {
            value.to_str().ok_or_else(|| VarError::NotUnicode(value.to_os_string()))
        } else {
            Err(VarError::NotPresent)
        }
    }
}
