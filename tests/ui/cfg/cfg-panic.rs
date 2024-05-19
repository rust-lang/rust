//@ build-pass
//@ compile-flags: -C panic=unwind
//@ needs-unwind

#[cfg(panic = "abort")]
pub fn bad() -> i32 { }

#[cfg(not(panic = "unwind"))]
pub fn bad() -> i32 { }

#[cfg(panic = "unwind")]
pub fn main() { }
