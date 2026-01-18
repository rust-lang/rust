// Auxiliary crate that adds a private item in rpass2.
// The SVH should remain stable, so dependents should not rebuild.

//@[rpass1] compile-flags: -Z query-dep-graph -Z stable-crate-hash
//@[rpass2] compile-flags: -Z query-dep-graph -Z stable-crate-hash

#![crate_type = "rlib"]

pub fn public_fn() -> i32 {
    42
}

pub struct PublicStruct {
    pub field: i32,
}

#[cfg(rpass2)]
fn private_fn() -> i32 {
    100
}

#[cfg(rpass2)]
struct PrivateStruct {
    field: String,
}

#[cfg(rpass2)]
const PRIVATE_CONST: i32 = 999;
