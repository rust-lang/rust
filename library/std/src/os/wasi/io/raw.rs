//! WASI-specific extensions to general I/O primitives.

#![unstable(feature = "wasi_ext", issue = "71213")]

pub use crate::os::fd::raw::*;
