// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(catch_all_match_arms)]
#![allow(dead_code, unused_variable)]

enum Things {
    Foo,
    Bar,
    Baz
}

// Lint on enums only

fn test1() {
    let a = Some(5u);
    match a {
        Some(x) => {}
        _ => {} //~ ERROR: catch-all pattern in match
    }
}

fn test2() {
    let a = Foo;
    match a {
        Foo => {}
        Bar => {}
        _ => {} //~ ERROR: catch-all pattern in match
    }
}

// Do not lint on other types

fn test3() {
    let b = 5u;
    let a = match b {
        9u => {}
        _ => {}
    };
}

fn test4() {
    let b: (int, uint) = (5i, 8u);
    match b {
        (3, 4) => {}
        (3, _) => {}
        _ => {}
    };
}

fn main() {}

