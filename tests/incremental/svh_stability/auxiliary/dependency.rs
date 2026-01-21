//@ revisions:rpass1 rpass2 rpass3

// A dependency crate with:
// - A private function that changes (rpass1 -> rpass2)
// - A public non-inlinable function
// - A public inlinable function that changes (rpass2 -> rpass3)

// Private function - changes between rpass1 and rpass2
// This should NOT affect downstream crates' SVH
fn private_helper() -> i32 {
    #[cfg(rpass1)]
    { 1 }

    #[cfg(any(rpass2, rpass3))]
    { 2 }
}

// Public function that calls the private helper
// Its body is NOT inlined downstream (no #[inline])
pub fn public_function() -> i32 {
    private_helper() + 10
}

// Public inlinable function - changes between rpass2 and rpass3
// This SHOULD affect downstream crates' SVH
#[inline]
pub fn inlinable_function() -> i32 {
    #[cfg(any(rpass1, rpass2))]
    { 100 }

    #[cfg(rpass3)]
    { 200 }
}

// Public struct - unchanged across all revisions
pub struct Data {
    pub value: i32,
}

impl Data {
    pub fn new(v: i32) -> Self {
        Data { value: v }
    }
}
