// Auxiliary crate for testing span reproducibility in metadata.
// This crate exports items that embed spans in the metadata,
// which are then used by the dependent crate.

//@[rpass1] compile-flags: -Z query-dep-graph
//@[rpass2] compile-flags: -Z query-dep-graph
//@[rpass3] compile-flags: -Z query-dep-graph --remap-path-prefix={{src-base}}=/remapped

#![crate_type = "rlib"]

/// A struct with spans in its definition.
pub struct SpannedStruct {
    pub field1: u32,
    pub field2: String,
}

/// A generic function whose span is embedded in metadata.
#[inline(always)]
pub fn generic_fn<T: Default>() -> T {
    T::default()
}

/// A macro that will have its expansion span in metadata.
#[macro_export]
macro_rules! span_macro {
    ($e:expr) => {
        $e + 1
    };
}

/// Trait with associated types to test span handling.
pub trait SpannedTrait {
    type Output;
    fn process(&self) -> Self::Output;
}

impl SpannedTrait for u32 {
    type Output = u32;
    fn process(&self) -> Self::Output {
        *self * 2
    }
}

/// Function with panic that embeds span in metadata.
#[inline(always)]
pub fn might_panic(x: u32) -> u32 {
    if x == 0 {
        panic!("x cannot be zero");
    }
    x
}
