// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

use std::default::Default;

pub struct X<T> {
    a: T,
}

// reordering these bounds stops the ICE
//
// nmatsakis: This test used to have the bounds Default + PartialEq +
// Default, but having duplicate bounds became illegal.
impl<T: Default + PartialEq> Default for X<T> {
    fn default() -> X<T> {
        X { a: Default::default() }
    }
}

macro_rules! constants {
    () => {
        let _ : X<isize> = Default::default();
    }
}

pub fn main() {
    constants!();
}
