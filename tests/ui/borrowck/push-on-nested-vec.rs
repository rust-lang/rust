//! Regression test for <https://github.com/rust-lang/rust/issues/3991>.
//! Test borrowck doesn't complain on nested vector mutable reference.
//@ check-pass

#![allow(dead_code)]

struct HasNested {
    nest: Vec<Vec<isize> > ,
}

impl HasNested {
    fn method_push_local(&mut self) {
        self.nest[0].push(0);
    }
}

pub fn main() {}
