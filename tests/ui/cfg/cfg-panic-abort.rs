//@ build-pass
//@ compile-flags: -C panic=abort
//@ no-prefer-dynamic
//@ ignore-backends: gcc

#[cfg(panic = "unwind")]
pub fn bad() -> i32 { }

#[cfg(not(panic = "abort"))]
pub fn bad() -> i32 { }

#[cfg(panic = "abort")]
pub fn main() { }
