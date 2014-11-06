// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Enum {
    Foo { foo: uint },
    Bar { bar: uint }
}

fn fun1(e1: &Enum, e2: &Enum) -> uint {
    match (e1, e2) {
        (&Foo { foo: _ }, &Foo { foo: _ }) => 0,
        (&Foo { foo: _ }, &Bar { bar: _ }) => 1,
        (&Bar { bar: _ }, &Bar { bar: _ }) => 2,
        (&Bar { bar: _ }, &Foo { foo: _ }) => 3,
    }
}

fn fun2(e1: &Enum, e2: &Enum) -> uint {
    match (e1, e2) {
        (&Foo { foo: _ }, &Foo { foo: _ }) => 0,
        (&Foo { foo: _ }, _              ) => 1,
        (&Bar { bar: _ }, &Bar { bar: _ }) => 2,
        (&Bar { bar: _ }, _              ) => 3,
    }
}

pub fn main() {
    let foo = Foo { foo: 1 };
    let bar = Bar { bar: 1 };

    assert_eq!(fun1(&foo, &foo), 0);
    assert_eq!(fun1(&foo, &bar), 1);
    assert_eq!(fun1(&bar, &bar), 2);
    assert_eq!(fun1(&bar, &foo), 3);

    assert_eq!(fun2(&foo, &foo), 0);
    assert_eq!(fun2(&foo, &bar), 1); // fun2 returns 0
    assert_eq!(fun2(&bar, &bar), 2);
    assert_eq!(fun2(&bar, &foo), 3); // fun2 returns 2
}
