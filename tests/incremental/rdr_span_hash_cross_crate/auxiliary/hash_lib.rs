// Auxiliary crate for testing span hash stability across crate boundaries.
// Changes to span encoding should not affect dependent crate hashes
// unless the actual source location changes.

//@[rpass1] compile-flags: -Z query-dep-graph
//@[rpass2] compile-flags: -Z query-dep-graph
//@[rpass3] compile-flags: -Z query-dep-graph

#![crate_type = "rlib"]

// NOTE: Do not change the line numbers of these functions between revisions!
// The test relies on spans staying at the same file:line:column locations.

#[inline(always)]
pub fn stable_span_fn() -> u32 {
    // This comment is here to ensure the function body has some content
    42
}

#[inline(always)]
pub fn generic_stable<T: Default + std::ops::Add<Output = T>>(x: T) -> T {
    x + T::default()
}

#[derive(Debug, Clone, PartialEq)]
pub struct StableStruct {
    pub value: u32,
}

impl StableStruct {
    pub fn new(value: u32) -> Self {
        Self { value }
    }
}
