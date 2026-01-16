// Auxiliary crate for testing that adding private functions
// does not cause dependent crates to rebuild.

#![crate_name = "dep"]
#![crate_type = "rlib"]

// Public API - unchanged across all revisions
pub fn public_fn(x: u32) -> u32 {
    x + 1
}

pub struct PublicStruct {
    pub value: u32,
}

// rpass2: Add a new private function
#[cfg(any(rpass2, rpass3))]
fn new_private_fn() -> u32 {
    42
}

// rpass3: Add another private function
#[cfg(rpass3)]
fn another_private_fn(x: u32) -> u32 {
    x * 2
}

// rpass3: Add a private struct
#[cfg(rpass3)]
struct PrivateStruct {
    _field: u32,
}
