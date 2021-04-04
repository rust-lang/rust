//! Various extensions traits for Chalk types.

use crate::{Interner, Ty, TyKind};

pub trait TyExt {
    fn is_unit(&self) -> bool;
}

impl TyExt for Ty {
    fn is_unit(&self) -> bool {
        matches!(self.kind(&Interner), TyKind::Tuple(0, _))
    }
}
