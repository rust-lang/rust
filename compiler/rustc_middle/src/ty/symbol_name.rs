use std::{fmt, str};

use crate::ty::TyCtxt;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, TyEncodable, HashStable)]
pub struct SymbolName<'tcx> {
    /// `&str` gives a consistent ordering, which ensures reproducible builds.
    pub name: &'tcx str,
}

impl<'tcx> SymbolName<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, name: &str) -> SymbolName<'tcx> {
        SymbolName {
            name: unsafe { str::from_utf8_unchecked(tcx.arena.alloc_slice(name.as_bytes())) },
        }
    }
}

impl<'tcx> fmt::Display for SymbolName<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.name, fmt)
    }
}

impl<'tcx> fmt::Debug for SymbolName<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.name, fmt)
    }
}
