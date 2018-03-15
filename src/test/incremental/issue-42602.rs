// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #42602. It used to be that we had
// a dep-graph like
//
//     typeck(foo) -> FnOnce -> typeck(bar)
//
// This was fixed by improving the resolution of the `FnOnce` trait
// selection node.

// revisions:cfail1
// compile-flags:-Zquery-dep-graph

#![feature(rustc_attrs)]

fn main() {
    a::foo();
    b::bar();
}

mod a {
    #[rustc_if_this_changed(HirBody)]
    pub fn foo() {
        let x = vec![1, 2, 3];
        let v = || ::std::mem::drop(x);
        v();
    }
}

mod b {
    #[rustc_then_this_would_need(TypeckTables)] //[cfail1]~ ERROR no path
    pub fn bar() {
        let x = vec![1, 2, 3];
        let v = || ::std::mem::drop(x);
        v();
    }
}
