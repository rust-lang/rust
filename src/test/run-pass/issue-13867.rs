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
        (FooUint(..), 'a'..'z') => 1,
        (FooNullary, 'x') => 2,
        _ => 0
    };
    assert_eq!(r, 0);

    let r = match (FooUint(0), 'a') {
        (FooUint(1), 'a'..'z') => 1,
        (FooUint(..), 'x') => 2,
        (FooNullary, 'a') => 3,
        _ => 0
    };
    assert_eq!(r, 0);

    let r = match ('a', FooUint(0)) {
        ('a'..'z', FooUint(1)) => 1,
        ('x', FooUint(..)) => 2,
        ('a', FooNullary) => 3,
        _ => 0
    };
    assert_eq!(r, 0);

    let r = match ('a', 'a') {
        ('a'..'z', 'b') => 1,
        ('x', 'a'..'z') => 2,
        _ => 0
    };
    assert_eq!(r, 0);

    let r = match ('a', 'a') {
        ('a'..'z', 'b') => 1,
        ('x', 'a'..'z') => 2,
        ('a', 'a') => 3,
        _ => 0
    };
    assert_eq!(r, 3);
}
