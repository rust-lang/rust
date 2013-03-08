// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #1112
// Alignment of interior pointers to dynamic-size types

struct X<T> {
    a: T,
    b: u8,
    c: bool,
    d: u8,
    e: u16,
    f: u8,
    g: u8
}

pub fn main() {
    let x: X<int> = X {
        a: 12345678,
        b: 9u8,
        c: true,
        d: 10u8,
        e: 11u16,
        f: 12u8,
        g: 13u8
    };
    bar(x);
}

fn bar<T>(x: X<T>) {
    fail_unless!(x.b == 9u8);
    fail_unless!(x.c == true);
    fail_unless!(x.d == 10u8);
    fail_unless!(x.e == 11u16);
    fail_unless!(x.f == 12u8);
    fail_unless!(x.g == 13u8);
}
