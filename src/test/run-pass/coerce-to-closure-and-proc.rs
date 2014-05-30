// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn id<T>(x: T) -> T {
    x
}

#[deriving(PartialEq, Show)]
struct Foo<T>(T);

#[deriving(PartialEq, Show)]
enum Bar<T> {
    Bar(T)
}

pub fn main() {
    let f: |int| -> int = id;
    assert_eq!(f(5), 5);

    let f: proc(int) -> int = id;
    assert_eq!(f(5), 5);

    let f: |int| -> Foo<int> = Foo;
    assert_eq!(f(5), Foo(5));

    let f: proc(int) -> Foo<int> = Foo;
    assert_eq!(f(5), Foo(5));

    let f: |int| -> Bar<int> = Bar;
    assert_eq!(f(5), Bar(5));

    let f: proc(int) -> Bar<int> = Bar;
    assert_eq!(f(5), Bar(5));

    let f: |int| -> Option<int> = Some;
    assert_eq!(f(5), Some(5));

    let f: proc(int) -> Option<int> = Some;
    assert_eq!(f(5), Some(5));
}
