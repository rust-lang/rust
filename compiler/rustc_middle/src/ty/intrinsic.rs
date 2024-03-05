use rustc_span::{def_id::DefId, Symbol};

use super::TyCtxt;

#[derive(Copy, Clone, Debug, Decodable, Encodable, HashStable)]
pub struct IntrinsicDef {
    pub name: Symbol,
    /// Whether the intrinsic has no meaningful body and all backends need to shim all calls to it.
    pub must_be_overridden: bool,
}

impl TyCtxt<'_> {
    pub fn is_intrinsic(self, def_id: DefId, name: Symbol) -> bool {
        let Some(i) = self.intrinsic(def_id) else { return false };
        i.name == name
    }
}
