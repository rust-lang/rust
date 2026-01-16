// Auxiliary crate for testing RDR span stability.
//
// This crate has private items whose spans change between revisions,
// but the public API remains identical. The dependent crate should
// NOT be rebuilt when only private span-affecting changes occur.
//
// rpass1: Initial state
// rpass2: Added blank lines before private fn (shifts all BytePos values)
// rpass3: Added comments inside private fn (changes internal spans)
// rpass4: Changed private fn body (implementation change, not just spans)

#![crate_name = "dep"]
#![crate_type = "rlib"]

// ============================================================
// PUBLIC API - This should remain stable across all revisions
// ============================================================

/// Public struct - its definition spans should be stable.
pub struct PublicStruct {
    pub value: u32,
}

impl PublicStruct {
    /// Public constructor - calls private helper internally.
    pub fn new(v: u32) -> Self {
        Self { value: private_transform(v) }
    }

    /// Public method - depends on private implementation.
    pub fn doubled(&self) -> u32 {
        private_double(self.value)
    }
}

/// Public function that uses private helpers.
pub fn public_compute(x: u32) -> u32 {
    let a = private_transform(x);
    let b = private_double(a);
    private_combine(a, b)
}

/// Public trait with default implementation using private fn.
pub trait PublicTrait {
    fn compute(&self) -> u32;

    fn with_default(&self) -> u32 {
        private_transform(self.compute())
    }
}

impl PublicTrait for u32 {
    fn compute(&self) -> u32 {
        *self
    }
}

// ============================================================
// PRIVATE IMPLEMENTATION - Changes here affect spans
// ============================================================

// rpass2+: These blank lines shift BytePos values of everything below.
// This simulates developers adding whitespace or reformatting code.
#[cfg(any(rpass2, rpass3, rpass4))]
const _BLANK_LINES_MARKER: () = ();




// End of blank lines section

/// Private helper function.
#[cfg(rpass1)]
fn private_transform(x: u32) -> u32 {
    x.wrapping_add(1)
}

/// Private helper function - with shifted spans in rpass2.
#[cfg(rpass2)]
fn private_transform(x: u32) -> u32 {
    x.wrapping_add(1)
}

/// Private helper function - with comments in rpass3.
#[cfg(rpass3)]
fn private_transform(x: u32) -> u32 {
    // Adding a comment here changes internal spans
    // but should not affect the dependent crate.
    x.wrapping_add(1)
}

/// Private helper function - with comments in rpass4.
#[cfg(rpass4)]
fn private_transform(x: u32) -> u32 {
    // Same comments as rpass3
    // but should not affect the dependent crate.
    x.wrapping_add(1)
}

/// Private double function.
#[cfg(any(rpass1, rpass2, rpass3))]
fn private_double(x: u32) -> u32 {
    x * 2
}

/// Private double function - changed implementation in rpass4.
#[cfg(rpass4)]
fn private_double(x: u32) -> u32 {
    x << 1  // Same result, different implementation
}

/// Private combiner function.
fn private_combine(a: u32, b: u32) -> u32 {
    a.wrapping_add(b)
}

// rpass2+: Additional private module to shift spans further.
#[cfg(any(rpass2, rpass3, rpass4))]
mod private_module {
    #[allow(dead_code)]
    pub fn unused_fn() -> u32 { 42 }
}
