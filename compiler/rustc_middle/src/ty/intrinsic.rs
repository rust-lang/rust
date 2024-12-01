use rustc_macros::{Decodable, Encodable, HashStable};
use rustc_span::Symbol;
use rustc_span::def_id::DefId;

use super::TyCtxt;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Decodable, Encodable, HashStable)]
pub enum IntrinsicKind {
    /// The intrinsic has no meaningful body and all backends need to shim all calls to it.
    MustBeOverridden,
    /// The intrinsic lowers to MIR, so does not need to be considered by backends.
    LowersToMir,
    /// The intrinsic has a meaningful body usable by backends that don't need something special.
    HasFallback,
}

#[derive(Copy, Clone, Debug, Decodable, Encodable, HashStable)]
pub struct IntrinsicDef {
    pub name: Symbol,
    /// Describes how the intrinsic is expected to be handled, based on its definition.
    pub kind: IntrinsicKind,
    /// Whether the intrinsic can be invoked from stable const fn
    pub const_stable: bool,
}

impl IntrinsicDef {
    pub fn must_be_overridden(self) -> bool {
        self.kind == IntrinsicKind::MustBeOverridden
    }

    pub fn has_fallback(self) -> bool {
        self.kind == IntrinsicKind::HasFallback
    }

    pub fn lowers_to_mir(self) -> bool {
        self.kind == IntrinsicKind::LowersToMir
    }
}

impl TyCtxt<'_> {
    pub fn is_intrinsic(self, def_id: DefId, name: Symbol) -> bool {
        let Some(i) = self.intrinsic(def_id) else { return false };
        i.name == name
    }
}
