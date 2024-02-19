use rustc_span::{def_id::DefId, Symbol};

use super::TyCtxt;

impl TyCtxt<'_> {
    pub fn is_intrinsic(self, def_id: DefId, name: Symbol) -> bool {
        let Some(i) = self.intrinsic(def_id) else { return false };
        i == name
    }
}
