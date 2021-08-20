//! WASI-specific extensions to general I/O primitives.

#![deny(unsafe_op_in_unsafe_fn)]
#![unstable(feature = "wasi_ext", issue = "71213")]

mod fd;
mod raw;

#[unstable(feature = "wasi_ext", issue = "71213")]
pub use fd::*;
#[unstable(feature = "wasi_ext", issue = "71213")]
pub use raw::*;
