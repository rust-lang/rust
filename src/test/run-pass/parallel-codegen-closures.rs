// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests parallel codegen - this can fail if the symbol for the anonymous
// closure in `sum` pollutes the second codegen unit from the first.

// ignore-bitrig
// compile-flags: -C codegen_units=2

#![feature(core)]
#![feature(iter_arith)]

mod a {
    fn foo() {
        let x = ["a", "bob", "c"];
        let len: usize = x.iter().map(|s| s.len()).sum();
    }
}

mod b {
    fn bar() {
        let x = ["a", "bob", "c"];
        let len: usize = x.iter().map(|s| s.len()).sum();
    }
}

fn main() {
}
