// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {

    // Note: here we do not have any type annotations
    // but we do express conflicting requirements:

    let mut v = ~[~[0]];
    let mut w = ~[~[0]];
    let mut x = ~[~[0]];

    fn f(&&v: ~[~[int]]) {
        v[0] = ~[3]
    }

    fn g(&&v: ~[const ~[const int]]) {
    }

    fn h(&&v: ~[~[int]]) {
        v[0] = ~[3]
    }

    fn i(&&v: ~[~[const int]]) {
        v[0] = ~[3]
    }

    fn j(&&v: ~[~[const int]]) {
    }

    f(v);
    g(v);
    h(v); //~ ERROR (values differ in mutability)
    i(v); //~ ERROR (values differ in mutability)
    j(v); //~ ERROR (values differ in mutability)

    f(w); //~ ERROR (values differ in mutability)
    g(w);
    h(w);
    i(w); //~ ERROR (values differ in mutability)
    j(w); //~ ERROR (values differ in mutability)

    // Note that without adding f() or h() to the mix, it is valid for
    // x to have the type ~[~[const int]], and thus we can safely
    // call g() and i() but not j():
    g(x);
    i(x);
    j(x); //~ ERROR (values differ in mutability)
}
