// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #48132. This was failing due to problems around
// the projection caching and dropck type enumeration.

// run-pass

#![feature(nll)]
#![allow(warnings)]

struct Inner<I, V> {
    iterator: I,
    item: V,
}

struct Outer<I: Iterator> {
    inner: Inner<I, I::Item>,
}

fn outer<I>(iterator: I) -> Outer<I>
where I: Iterator,
      I::Item: Default,
{
    Outer {
        inner: Inner {
            iterator: iterator,
            item: Default::default(),
        }
    }
}

fn main() {
    outer(std::iter::once(&1).cloned());
}
