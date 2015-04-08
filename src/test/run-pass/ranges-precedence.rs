// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the precedence of ranges is correct



struct Foo {
    foo: usize,
}

impl Foo {
    fn bar(&self) -> usize { 5 }
}

fn main() {
    let x = 1+3..4+5;
    assert!(x == (4..9));

    let x = 1..4+5;
    assert!(x == (1..9));

    let x = 1+3..4;
    assert!(x == (4..4));

    let a = Foo { foo: 3 };
    let x = a.foo..a.bar();
    assert!(x == (3..5));

    let x = 1+3..;
    assert!(x == (4..));
    let x = ..1+3;
    assert!(x == (..4));

    let a = &[0, 1, 2, 3, 4, 5, 6];
    let x = &a[1+1..2+2];
    assert!(x == &a[2..4]);
    let x = &a[..1+2];
    assert!(x == &a[..3]);
    let x = &a[1+2..];
    assert!(x == &a[3..]);

    for _i in 2+4..10-3 {}

    let i = 42;
    for _ in 1..i {}
    for _ in 1.. { break; }

    let x = [1]..[2];
    assert!(x == (([1])..([2])));

    let y = ..;
    assert!(y == (..));
}
