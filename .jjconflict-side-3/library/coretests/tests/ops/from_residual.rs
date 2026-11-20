//! Regression test that Option and ControlFlow can have downstream FromResidual impls.
//! cc https://github.com/rust-lang/rust/issues/99940,
//! This does NOT test that issue in general; Option and ControlFlow's FromResidual
//! impls in core were changed to not be affected by that issue.

use core::ops::{ControlFlow, FromResidual};

struct Local;

impl<T> FromResidual<Local> for Option<T> {
    fn from_residual(_: Local) -> Option<T> {
        unimplemented!()
    }
}

impl<B, C> FromResidual<Local> for ControlFlow<B, C> {
    fn from_residual(_: Local) -> ControlFlow<B, C> {
        unimplemented!()
    }
}

impl<T, E> FromResidual<Local> for Result<T, E> {
    fn from_residual(_: Local) -> Result<T, E> {
        unimplemented!()
    }
}
