//! WASI-specific extensions to general I/O primitives.

#![stable(feature = "io_safety_wasi", since = "1.65.0")]

#[stable(feature = "io_safety_wasi", since = "1.65.0")]
#[allow(clippy::useless_attribute)]
#[allow(incompatible_reexport_stability)]
// This re-export has its own stability.
pub use crate::os::fd::*;

// Tests for this module
#[cfg(test)]
mod tests;
