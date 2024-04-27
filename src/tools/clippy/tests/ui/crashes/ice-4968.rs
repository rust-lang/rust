// Test for https://github.com/rust-lang/rust-clippy/issues/4968

#![warn(clippy::unsound_collection_transmute)]
#![allow(clippy::transmute_undefined_repr)]

trait Trait {
    type Assoc;
}

use std::mem::{self, ManuallyDrop};

#[allow(unused)]
fn func<T: Trait>(slice: Vec<T::Assoc>) {
    unsafe {
        let _: Vec<ManuallyDrop<T::Assoc>> = mem::transmute(slice);
    }
}

fn main() {}
