// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #5886: a complex instance of issue #2687.

trait Iterator<A> {
    fn next(&mut self) -> Option<A>;
}

trait IteratorUtil<A>: Sized
{
    fn zip<B, U: Iterator<U>>(self, other: U) -> ZipIterator<Self, U>;
}

impl<A, T: Iterator<A>> IteratorUtil<A> for T {
    fn zip<B, U: Iterator<B>>(self, other: U) -> ZipIterator<T, U> {
    //~^ ERROR the requirement `U : Iterator<B>` appears on the impl method
        ZipIterator{a: self, b: other}
    }
}

struct ZipIterator<T, U> {
    a: T, b: U
}

fn main() {}
