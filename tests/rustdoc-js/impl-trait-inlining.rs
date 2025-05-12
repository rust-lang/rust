#![crate_name = "foo"]

pub struct TyCtxt;
pub struct DefId;
pub struct Symbol;

impl TyCtxt {
    pub fn has_attr(self, _did: impl Into<DefId>, _attr: Symbol) -> bool {
        unimplemented!();
    }
}
