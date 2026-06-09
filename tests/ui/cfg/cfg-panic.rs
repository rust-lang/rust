//@ build-pass
//@ compile-flags: -C panic=unwind
//@ needs-unwind
//@ reference: cfg.panic.def
//@ reference: cfg.panic.values

#[cfg(panic = "abort")]
pub fn bad() -> i32 { }

#[cfg(not(panic = "unwind"))]
pub fn bad() -> i32 { }

#[cfg(panic = "unwind")]
pub fn main() { }
