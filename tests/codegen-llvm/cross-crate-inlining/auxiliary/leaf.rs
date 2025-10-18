//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// This function *looks* like it contains a call, but that call will be optimized out by MIR
// optimizations.
pub fn leaf_fn() -> bool {
    Some(0).is_some()
}

// This function contains a call, even after MIR optimizations. It is only eligible for
// cross-crate-inlining with "always".
pub fn stem_fn() -> String {
    inner()
}

#[inline(never)]
fn inner() -> String {
    String::from("test")
}

// This function's optimized MIR contains a call, but it is to an intrinsic.
pub fn leaf_with_intrinsic(a: &[u64; 2], b: &[u64; 2]) -> bool {
    a == b
}
