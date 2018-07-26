// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that changing what a `type` points to does not go unnoticed
// by the variance analysis.

// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![allow(dead_code)]
#![allow(unused_variables)]

fn main() { }

struct Foo<T> {
    f: T
}

#[rustc_if_this_changed(Krate)]
type TypeAlias<T> = Foo<T>;

#[rustc_then_this_would_need(ItemVariances)] //~ ERROR OK
struct Use<T> {
    x: TypeAlias<T>
}
