//! Sanitizers support for the Rust compiler.
//!
//! This crate contains the source code for providing support for the sanitizers to the Rust
//! compiler.

// tidy-alphabetical-start
#![feature(let_chains)]
// tidy-alphabetical-end

pub mod cfi;
pub mod kcfi;
