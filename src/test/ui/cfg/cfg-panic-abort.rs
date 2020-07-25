// build-pass
// compile-flags: -C panic=abort
// no-prefer-dynamic
#![feature(cfg_panic)]

#[cfg(panic = "unwind")]
pub fn bad() -> i32 { }

#[cfg(not(panic = "abort"))]
pub fn bad() -> i32 { }

#[cfg(panic = "some_imaginary_future_panic_handler")]
pub fn bad() -> i32 { }

#[cfg(panic = "abort")]
pub fn main() { }
