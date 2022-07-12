#![allow(deprecated)]
// This test is adapted from https://github.com/rust-lang/miri/issues/1340#issue-600900312.

fn main() {
    // Deliberately using `mem::uninitialized` to make sure that despite all the mitigations, we consider this UB.
    let _val: f32 = unsafe { std::mem::uninitialized() };
    //~^ ERROR: constructing invalid value at .value: encountered uninitialized bytes, but expected initialized bytes
}
