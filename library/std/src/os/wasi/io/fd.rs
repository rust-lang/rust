//! Owned and borrowed file descriptors.

#![unstable(feature = "wasi_ext", issue = "71213")]

// Tests for this module
#[cfg(test)]
mod tests;

pub use super::super::super::super::fd::*;
