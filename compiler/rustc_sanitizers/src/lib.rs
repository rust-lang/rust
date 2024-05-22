#![feature(let_chains)]
//! Sanitizers support for the Rust compiler.
//!
//! This crate contains the source code for providing support for the sanitizers to the Rust
//! compiler.
pub mod cfi;
pub mod kcfi;
