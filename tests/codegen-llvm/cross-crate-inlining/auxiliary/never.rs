//@ compile-flags: -Copt-level=3 -Zcross-crate-inline-threshold=never

#![crate_type = "lib"]

// This function *looks* like it contains a call, but that call will be optimized out by MIR
// optimizations.
pub fn leaf_fn() -> String {
    String::new()
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
