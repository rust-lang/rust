//@ run-pass

#![feature(step_trait)]

use std::iter::Step;

#[cfg(target_pointer_width = "16")]
fn main() {
    assert!(Step::steps_between(&0u32, &u32::MAX).1.is_none());
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
fn main() {
    assert!(Step::steps_between(&0u32, &u32::MAX).1.is_some());
}
