// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

// Here we expect a coherence conflict because, even though `i32` does
// not implement `Iterator`, we cannot rely on that negative reasoning
// due to the orphan rules. Therefore, `A::Item` may yet turn out to
// be `i32`.

pub trait Foo<P> {}

pub trait Bar {
    type Output: 'static;
}

impl Foo<i32> for i32 { } //~ ERROR E0119

impl<A:Iterator> Foo<A::Item> for A { }

fn main() {}
