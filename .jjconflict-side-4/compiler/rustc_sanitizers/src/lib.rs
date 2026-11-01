//! Sanitizers support for the Rust compiler.
//!
//! This crate contains the source code for providing support for the sanitizers to the Rust
//! compiler.

#![feature(extern_types)]

// tidy-alphabetical-start
// tidy-alphabetical-end

pub mod cfi;
pub mod ignorelist;
pub mod kcfi;
