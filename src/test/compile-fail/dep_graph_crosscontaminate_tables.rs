// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the `Tables` nodes for impl items are independent from
// one another.

// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]

struct Foo {
    x: u8
}

impl Foo {
    // Changing the item `new`...
    #[rustc_if_this_changed(HirBody)]
    fn new() -> Foo {
        Foo { x: 0 }
    }

    // ...should not cause us to recompute the tables for `with`!
    #[rustc_then_this_would_need(Tables)] //~ ERROR no path
    fn with(x: u8) -> Foo {
        Foo { x: x }
    }
}

fn main() {
    let f = Foo::new();
    let g = Foo::with(22);
    assert_eq!(f.x, g.x - 22);
}
