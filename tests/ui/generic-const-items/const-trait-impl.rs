// known-bug: #110395
// FIXME check-pass

// Test that we can call methods from const trait impls inside of generic const items.

#![feature(generic_const_items, const_trait_impl)]
#![allow(incomplete_features)]
#![crate_type = "lib"]

// FIXME(generic_const_items, effects): Introduce `const` bounds to make this work.
const CREATE<T: Create>: T = T::create();

pub const K0: i32 = CREATE::<i32>;
pub const K1: i32 = CREATE; // arg inferred

#[const_trait]
trait Create {
    fn create() -> Self;
}

impl const Create for i32 {
    fn create() -> i32 {
        4096
    }
}
