//! Traits, helpers, and type definitions for core I/O functionality.

#![unstable(feature = "alloc_io", issue = "none")]

pub use core::io::*;

#[cfg(target_has_atomic_load_store = "ptr")]
mod error;
