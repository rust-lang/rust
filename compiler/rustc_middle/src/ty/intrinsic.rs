use std::fmt::Display;

use rustc_macros::{Decodable, Encodable, StableHash};
use rustc_span::Symbol;
use rustc_span::def_id::DefId;

use super::TyCtxt;

#[derive(Copy, Clone, Debug, Decodable, Encodable, StableHash, PartialEq, Eq)]
pub enum IntrinsicName {
    RustIntrinsic(Symbol),
    BackendIntrinsic { llvm: Symbol },
}

impl Display for IntrinsicName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntrinsicName::RustIntrinsic(i) => write!(f, "{i}"),
            IntrinsicName::BackendIntrinsic { llvm } => write!(f, "{llvm}"),
        }
    }
}

#[derive(Copy, Clone, Debug, Decodable, Encodable, StableHash)]
pub struct IntrinsicDef {
    pub name: IntrinsicName,
    /// Whether the intrinsic has no meaningful body and all backends need to shim all calls to it.
    pub must_be_overridden: bool,
    /// Whether the intrinsic can be invoked from stable const fn (`#[rustc_intrinsic_const_stable_indirect]`).
    pub const_stable_indirect: bool,
}

impl TyCtxt<'_> {
    pub fn is_rust_intrinsic(self, def_id: DefId, name: Symbol) -> bool {
        let Some(i) = self.intrinsic(def_id) else { return false };
        matches!(i.name, IntrinsicName::RustIntrinsic(n) if n == name)
    }
}
