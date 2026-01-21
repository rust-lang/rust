// Test library for RDR stable-crate-hash testing.
// This crate exports various items that embed spans in metadata.

#![crate_name = "rdr_test_lib"]
#![crate_type = "rlib"]

/// A public struct with spans in its definition.
pub struct TestStruct {
    pub field: u32,
}

/// A generic function whose span is embedded in metadata.
#[inline(always)]
pub fn generic_fn<T: Default>() -> T {
    T::default()
}

/// A function with panic (embeds panic location span).
pub fn might_panic(x: u32) -> u32 {
    assert!(x > 0, "x must be positive");
    x
}

/// Trait for testing trait impl spans.
pub trait TestTrait {
    fn process(&self) -> u32;
}

impl TestTrait for u32 {
    fn process(&self) -> u32 {
        *self * 2
    }
}

/// Macro that embeds expansion spans.
#[macro_export]
macro_rules! test_macro {
    ($e:expr) => {
        $e + 1
    };
}
