// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-pretty

// Tests correct copying of heap closures' environments.

fn bar<T: Copy>(x: T) -> (T, T) {
    (copy x, x)
}
fn foo(x: ~fn:Copy()) -> (~fn(), ~fn()) {
    bar(x)
}
fn main() {
    let v = ~[~[1,2,3],~[4,5,6]]; // shouldn't get double-freed
    let (f1,f2) = do foo {
        assert!(v.len() == 2);
    };
    f1();
    f2();
}
