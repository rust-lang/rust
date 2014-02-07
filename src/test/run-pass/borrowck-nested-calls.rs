// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test FIXME (#5074) nested method calls

// Test that (safe) nested calls with `&mut` receivers are permitted.

struct Foo {a: uint, b: uint}

impl Foo {
    pub fn inc_a(&mut self, v: uint) { self.a += v; }

    pub fn next_b(&mut self) -> uint {
        let b = self.b;
        self.b += 1;
        b
    }
}

pub fn main() {
    let mut f = Foo {a: 22, b: 23};
    f.inc_a(f.next_b());
    assert_eq!(f.a, 22+23);
    assert_eq!(f.b, 24);
}
