//! WASI-specific extensions to general I/O primitives.

#![deny(unsafe_op_in_unsafe_fn)]
#![stable(feature = "io_safety_wasi", since = "1.65.0")]

mod fd;
mod raw;

#[stable(feature = "io_safety_wasi", since = "1.65.0")]
pub use fd::*;
#[stable(feature = "io_safety_wasi", since = "1.65.0")]
pub use raw::*;
