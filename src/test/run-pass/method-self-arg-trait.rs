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

static mut COUNT: u64 = 1;

struct Foo;

trait Bar {
    fn foo1(&self);
    fn foo2(self);
    fn foo3(self: Box<Self>);

    fn bar1(&self) {
        unsafe { COUNT *= 7; }
    }
    fn bar2(self) {
        unsafe { COUNT *= 11; }
    }
    fn bar3(self: Box<Self>) {
        unsafe { COUNT *= 13; }
    }
}

impl Bar for Foo {
    fn foo1(&self) {
        unsafe { COUNT *= 2; }
    }

    fn foo2(self) {
        unsafe { COUNT *= 3; }
    }

    fn foo3(self: Box<Foo>) {
        unsafe { COUNT *= 5; }
    }
}

impl Foo {
    fn baz(self) {
        unsafe { COUNT *= 17; }
        // Test internal call.
        Bar::foo1(&self);
        Bar::foo2(self);
        Bar::foo3(box self);

        Bar::bar1(&self);
        Bar::bar2(self);
        Bar::bar3(box self);
    }
}

fn main() {
    let x = Foo;
    // Test external call.
    Bar::foo1(&x);
    Bar::foo2(x);
    Bar::foo3(box x);

    Bar::bar1(&x);
    Bar::bar2(x);
    Bar::bar3(box x);

    x.baz();

    unsafe { assert!(COUNT == 2u64*2*3*3*5*5*7*7*11*11*13*13*17); }
}
