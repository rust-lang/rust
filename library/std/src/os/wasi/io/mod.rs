//! WASI-specific extensions to general I/O primitives.

#![unstable(feature = "wasi_ext", issue = "71213")]

#[stable(feature = "io_safety", since = "1.63.0")]
pub use crate::os::fd::*;
