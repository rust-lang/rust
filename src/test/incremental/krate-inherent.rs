// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: cfail1 cfail2
// compile-flags: -Z query-dep-graph
// compile-pass

#![allow(warnings)]
#![feature(rustc_attrs)]
#![rustc_partition_reused(module="krate_inherent-x", cfg="cfail2")]
#![crate_type = "rlib"]

pub mod x {
    pub struct Foo;
    impl Foo {
        pub fn foo(&self) { }
    }

    pub fn method() {
        let x: Foo = Foo;
        x.foo(); // inherent methods used to add an edge from Krate
    }
}

#[cfg(cfail1)]
pub fn bar() { } // remove this unrelated fn in cfail2, which should not affect `x::method`

