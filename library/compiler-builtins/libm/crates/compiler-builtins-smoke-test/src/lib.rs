//! Fake compiler-builtins crate
//!
//! This is used to test that we can source import `libm` into the compiler-builtins crate.
//! Additionally, it provides a `#[no_mangle]` C API that can be easier to inspect than the
//! default `.rlib`.

#![compiler_builtins]
#![feature(core_intrinsics)]
#![feature(compiler_builtins)]
#![feature(f16)]
#![feature(f128)]
#![allow(internal_features)]
#![no_std]

mod math;
// Required for macro paths.
use math::libm::support;
