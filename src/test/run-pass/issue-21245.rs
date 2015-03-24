// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #21245. Check that we are able to infer
// the types in these examples correctly. It used to be that
// insufficient type propagation caused the type of the iterator to be
// incorrectly unified with the `*const` type to which it is coerced.

// pretty-expanded FIXME #23616

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
    match IntoIterator::into_iter(v.iter()) {
        mut iter => {
            loop {
                match ::std::iter::Iterator::next(&mut iter) {
                    ::std::option::Option::Some(x) => {
                        unsafe { ptr::read(x); }
                    },
                    ::std::option::Option::None => break
                }
            }
        }
    }
}

fn desugared_for_loop_good<T>(v: Vec<T>) {
    match v.iter().into_iter() {
        mut iter => {
            loop {
                match ::std::iter::Iterator::next(&mut iter) {
                    ::std::option::Option::Some(x) => {
                        unsafe { ptr::read(x); }
                    },
                    ::std::option::Option::None => break
                }
            }
        }
    }
}

fn main() {}
