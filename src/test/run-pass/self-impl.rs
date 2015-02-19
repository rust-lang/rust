// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can use `Self` types in impls in the expected way.

#![allow(unknown_features)]
#![feature(box_syntax)]

struct Foo;

// Test uses on inherent impl.
impl Foo {
    fn foo(_x: Self, _y: &Self, _z: Box<Self>) -> Self {
        Foo
    }
}

// Test uses when implementing a trait and with a type parameter.
pub struct Baz<X> {
    pub f: X,
}

trait Bar<X> {
    fn bar(x: Self, y: &Self, z: Box<Self>) -> Self;
    fn dummy(&self, x: X) { }
}

impl Bar<int> for Box<Baz<int>> {
    fn bar(_x: Self, _y: &Self, _z: Box<Self>) -> Self {
        box Baz { f: 42 }
    }
}

fn main() {
    let _: Foo = Foo::foo(Foo, &Foo, box Foo);
    let _: Box<Baz<int>> = Bar::bar(box Baz { f: 42 },
                                    &box Baz { f: 42 },
                                    box box Baz { f: 42 });
}
