//! WASI-specific extensions to general I/O primitives.

#![stable(feature = "io_safety_wasi", since = "1.65.0")]

#[stable(feature = "io_safety_wasi", since = "1.65.0")]
pub use crate::os::fd::*;

// Tests for this module
#[cfg(test)]
mod tests;
