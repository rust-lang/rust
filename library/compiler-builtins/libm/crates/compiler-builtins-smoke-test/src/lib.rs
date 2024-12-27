//! Fake compiler-builtins crate
//!
//! This is used to test that we can source import `libm` into the compiler-builtins crate.

#![feature(core_intrinsics)]
#![allow(internal_features)]
#![no_std]

#[allow(dead_code)]
#[allow(clippy::all)] // We don't get `libm`'s list of `allow`s, so just ignore Clippy.
#[path = "../../../src/math/mod.rs"]
pub mod libm;
