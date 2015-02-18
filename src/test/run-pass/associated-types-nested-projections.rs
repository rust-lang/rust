// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can resolve nested projection types. Issue #20666.

use std::marker::MarkerTrait;
use std::slice;

trait Bound : MarkerTrait {}

impl<'a> Bound for &'a i32 {}

trait IntoIterator {
    type Iter: Iterator;

    fn into_iter(self) -> Self::Iter;
}

impl<'a, T> IntoIterator for &'a [T; 3] {
    type Iter = slice::Iter<'a, T>;

    fn into_iter(self) -> slice::Iter<'a, T> {
        self.iter()
    }
}

fn foo<X>(x: X) where
    X: IntoIterator,
    <<X as IntoIterator>::Iter as Iterator>::Item: Bound,
{
}

fn bar<T, I, X>(x: X) where
    T: Bound,
    I: Iterator<Item=T>,
    X: IntoIterator<Iter=I>,
{

}

fn main() {
    foo(&[0, 1, 2]);
    bar(&[0, 1, 2]);
}
