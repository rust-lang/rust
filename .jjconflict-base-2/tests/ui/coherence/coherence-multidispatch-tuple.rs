//@ check-pass
#![allow(unused_imports)]

use std::fmt::Debug;
use std::default::Default;

// Test that an impl for homogeneous pairs does not conflict with a
// heterogeneous pair.

trait MyTrait {
    fn get(&self) -> usize;
}

impl<T> MyTrait for (T,T) {
    fn get(&self) -> usize { 0 }
}

impl MyTrait for (usize,isize) {
    fn get(&self) -> usize { 0 }
}

fn main() {
}
