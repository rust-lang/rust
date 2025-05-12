//! Handle the conversion of different internal errors into a stable version.
//!
//! Currently we encode everything as [stable_mir::Error], which is represented as a string.

use rustc_middle::mir::interpret::AllocError;
use rustc_middle::ty::layout::LayoutError;

use crate::rustc_smir::{Stable, Tables};
use crate::stable_mir;

impl<'tcx> Stable<'tcx> for LayoutError<'tcx> {
    type T = stable_mir::Error;

    fn stable(&self, _tables: &mut Tables<'_>) -> Self::T {
        stable_mir::Error::new(format!("{self:?}"))
    }
}

impl<'tcx> Stable<'tcx> for AllocError {
    type T = stable_mir::Error;

    fn stable(&self, _tables: &mut Tables<'_>) -> Self::T {
        stable_mir::Error::new(format!("{self:?}"))
    }
}
