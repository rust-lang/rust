// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for type inference failure. In this case,
// `ptr::read` expects a `*T`, but the result from the call to
// `next.unwrap()` was a type variable that had not yet been resolved.
// Once pending obligations are resolved, result is `&T`, which can be
// coered to `*T`. But if we first unify the types, then the
// resolution of the pending trait obligations fails.

use std::ptr;

trait IntoIterator {
    type Iter: Iterator;

    fn into_iter(self) -> Self::Iter;
}

impl<I> IntoIterator for I where I: Iterator {
    type Iter = I;

    fn into_iter(self) -> I {
        self
    }
}

fn desugared_for_loop_bad<T>(v: Vec<T>) {
    let mut iter = IntoIterator::into_iter(v.iter());
    let next = Iterator::next(&mut iter);
    let x = next.unwrap();
    unsafe { ptr::read(x); }
}

fn main() {}
