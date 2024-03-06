//@ run-pass

#![feature(unsized_fn_params)]

use std::ops;
use std::ops::Index;

pub struct A;

impl ops::Index<str> for A {
    type Output = ();
    fn index(&self, _: str) -> &Self::Output {
        &()
    }
}

impl ops::IndexMut<str> for A {
    fn index_mut(&mut self, _: str) -> &mut Self::Output {
        panic!()
    }
}

fn main() {
    let a = A {};
    let s = String::new().into_boxed_str();
    assert_eq!(&(), a.index(*s));
}
