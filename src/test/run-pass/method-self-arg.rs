// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test method calls with self as an argument

static mut COUNT: uint = 1;

struct Foo;

impl Foo {
    fn foo(self, x: &Foo) {
        unsafe { COUNT *= 2; }
        // Test internal call.
        Foo::bar(&self);
        Foo::bar(x);

        Foo::baz(self);
        Foo::baz(*x);

        Foo::qux(box self);
        Foo::qux(box *x);
    }

    fn bar(&self) {
        unsafe { COUNT *= 3; }
    }

    fn baz(self) {
        unsafe { COUNT *= 5; }
    }

    fn qux(self: Box<Foo>) {
        unsafe { COUNT *= 7; }
    }
}

fn main() {
    let x = Foo;
    // Test external call.
    Foo::bar(&x);
    Foo::baz(x);
    Foo::qux(box x);

    x.foo(&x);

    unsafe { assert!(COUNT == 2u*3*3*3*5*5*5*7*7*7); }
}
