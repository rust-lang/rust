// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that certain pattern-match type errors are non-fatal

enum A {
    B(int, int),
    C(int, int, int),
    D
}

struct S {
    a: int
}

fn f(_c: char) {}

fn main() {
    match B(1, 2) {
        B(_, _, _) => (), //~ ERROR this pattern has 3 fields, but
        D(_) => (),       //~ ERROR this pattern has 1 field, but
        _ => ()
    }
    match 'c' {
        S { .. } => (),   //~ ERROR mismatched types: expected `char`, found a structure pattern

        _ => ()
    }
    f(true);            //~ ERROR mismatched types: expected `char`, found `bool`
}
