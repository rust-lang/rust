#![allow(deprecated, invalid_value)]
// This test is adapted from https://github.com/rust-lang/miri/issues/1340#issue-600900312.

fn main() {
    // Deliberately using `mem::uninitialized` to make sure that despite all the mitigations, we consider this UB.
    // The array avoids a `Scalar` layout which detects uninit without even doing validation.
    let _val: [f32; 1] = unsafe { std::mem::uninitialized() };
    //~^ ERROR: uninitialized
}
