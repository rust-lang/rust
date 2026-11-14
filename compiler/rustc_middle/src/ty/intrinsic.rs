use std::fmt::Display;

use rustc_errors::IntoDiagArg;
use rustc_macros::{Decodable, Encodable, StableHash};
use rustc_span::Symbol;
use rustc_span::def_id::DefId;

use super::TyCtxt;

/// The name of an intrinsic.
#[derive(Copy, Clone, Debug, Decodable, Encodable, StableHash, PartialEq, Eq)]
pub enum Intrinsic {
    /// A Rust intrinsic (common to all backends).
    Rustc(Symbol),
    /// A backend-specific intrinsic. For now, we only support llvm-specific intrinsics.
    Backend { llvm: Symbol },
}

impl Display for Intrinsic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Intrinsic::Rustc(i) => write!(f, "{i}"),
            Intrinsic::Backend { llvm } => write!(f, "{llvm}"),
        }
    }
}

impl IntoDiagArg for Intrinsic {
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> rustc_errors::DiagArgValue {
        self.to_string().into_diag_arg(path)
    }
}

#[derive(Copy, Clone, Debug, Decodable, Encodable, StableHash)]
pub struct IntrinsicDef {
    pub name: Intrinsic,
    /// Whether the intrinsic has no meaningful body and all backends need to shim all calls to it.
    pub must_be_overridden: bool,
    /// Whether the intrinsic can be invoked from stable const fn (`#[rustc_intrinsic_const_stable_indirect]`).
    pub const_stable_indirect: bool,
}

impl TyCtxt<'_> {
    pub fn is_rustc_intrinsic(self, def_id: DefId, name: Symbol) -> bool {
        let Some(i) = self.intrinsic(def_id) else { return false };
        matches!(i.name, Intrinsic::Rustc(n) if n == name)
    }
}
