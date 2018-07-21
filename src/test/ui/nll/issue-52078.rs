// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]
#![allow(unused_variables)]

// Regression test for #52078: we were failing to infer a relationship
// between `'a` and `'b` below due to inference variables introduced
// during the normalization process.
//
// compile-pass

struct Drain<'a, T: 'a> {
    _marker: ::std::marker::PhantomData<&'a T>,
}

trait Join {
    type Value;
    fn get(value: &mut Self::Value);
}

impl<'a, T> Join for Drain<'a, T> {
    type Value = &'a mut Option<T>;

    fn get<'b>(value: &'b mut Self::Value) {
    }
}

fn main() {
}
