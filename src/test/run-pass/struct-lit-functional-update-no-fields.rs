// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Eq,Clone)]
struct Foo<T> {
    bar: T,
    baz: T
}

pub fn main() {
    let foo = Foo {
        bar: 0,
        baz: 1
    };

    let foo_ = foo.clone();
    let foo = Foo { ..foo };
    assert_eq!(foo, foo_);

    let foo = Foo {
        bar: ~"one",
        baz: ~"two"
    };

    let foo_ = foo.clone();
    let foo = Foo { ..foo };
    assert_eq!(foo, foo_);
}
