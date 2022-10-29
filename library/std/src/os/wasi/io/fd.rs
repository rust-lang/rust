//! Owned and borrowed file descriptors.

#![stable(feature = "io_safety_wasi", since = "1.65.0")]

// Tests for this module
#[cfg(test)]
mod tests;

#[stable(feature = "io_safety_wasi", since = "1.65.0")]
pub use crate::os::fd::owned::*;
