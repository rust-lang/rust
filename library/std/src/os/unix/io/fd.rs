//! Owned and borrowed file descriptors.

// Tests for this module
#[cfg(test)]
mod tests;

#[stable(feature = "io_safety", since = "1.63.0")]
pub use crate::os::fd::owned::*;
