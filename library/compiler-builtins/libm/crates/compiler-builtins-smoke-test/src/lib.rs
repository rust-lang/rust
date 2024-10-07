//! Fake compiler-builtins crate
//!
//! This is used to test that we can source import `libm` into the compiler-builtins crate.

#![feature(core_intrinsics)]
#![allow(internal_features)]
#![allow(dead_code)]
#![no_std]

#[path = "../../../src/math/mod.rs"]
pub mod libm;
