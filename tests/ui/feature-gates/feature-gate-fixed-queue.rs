// This test ensures that the `fixed_queue` feature gate is required
#![allow(unused)]

// Attempt to use the feature without enabling it
fn main() {
    let _queue = FixedQueue::<i32, 3>::new(); //~ ERROR the type `FixedQueue` is unstable
}
