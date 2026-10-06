//! Regression test that Option and ControlFlow can have downstream TryFromBreak impls.
//! cc https://github.com/rust-lang/rust/issues/99940,
//! This does NOT test that issue in general; Option and ControlFlow's TryFromBreak
//! impls in core were changed to not be affected by that issue.

use core::ops::{ControlFlow, TryFromBreak};

struct Local;

impl<T> TryFromBreak<Local> for Option<T> {
    fn from_break(_: Local) -> Option<T> {
        unimplemented!()
    }
}

impl<B, C> TryFromBreak<Local> for ControlFlow<B, C> {
    fn from_break(_: Local) -> ControlFlow<B, C> {
        unimplemented!()
    }
}

impl<T, E> TryFromBreak<Local> for Result<T, E> {
    fn from_break(_: Local) -> Result<T, E> {
        unimplemented!()
    }
}
