// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #47206: ensure that impls which have where
// clauses don't silently fail.
//
// compile-pass

#![feature(generic_associated_types)]
//~^ WARNING the feature `generic_associated_types` is incomplete

trait Foo {
    type Out;
}

impl<T> Foo for Box<T> {
    type Out where T: Clone = T;
}

fn main() {}
