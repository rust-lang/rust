//! Owned and borrowed file descriptors.

#![unstable(feature = "io_safety", issue = "87074")]

// Tests for this module
#[cfg(test)]
mod tests;

pub use crate::os::fd::owned::*;
