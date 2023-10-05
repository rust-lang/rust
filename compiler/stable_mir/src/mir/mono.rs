use crate::{ty::Instance, DefId, Opaque};

#[derive(Clone, Debug)]
pub enum MonoItem {
    Fn(Instance),
    Static(DefId),
    GlobalAsm(Opaque),
}
