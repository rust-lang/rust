// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that codegen works correctly when there are multiple refutable
// patterns in match expression.

enum Foo {
    FooUint(uint),
    FooNullary,
}

fn main() {
    let r = match (FooNullary, 'a') {
        (FooUint(..), 'a'..'z') => 1i,
        (FooNullary, 'x') => 2i,
        _ => 0
    };
    assert_eq!(r, 0);

    let r = match (FooUint(0), 'a') {
        (FooUint(1), 'a'..'z') => 1i,
        (FooUint(..), 'x') => 2i,
        (FooNullary, 'a') => 3i,
        _ => 0
    };
    assert_eq!(r, 0);

    let r = match ('a', FooUint(0)) {
        ('a'..'z', FooUint(1)) => 1i,
        ('x', FooUint(..)) => 2i,
        ('a', FooNullary) => 3i,
        _ => 0
    };
    assert_eq!(r, 0);

    let r = match ('a', 'a') {
        ('a'..'z', 'b') => 1i,
        ('x', 'a'..'z') => 2i,
        _ => 0
    };
    assert_eq!(r, 0);

    let r = match ('a', 'a') {
        ('a'..'z', 'b') => 1i,
        ('x', 'a'..'z') => 2i,
        ('a', 'a') => 3i,
        _ => 0
    };
    assert_eq!(r, 3);
}
