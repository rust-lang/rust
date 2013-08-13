// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can coerce an `@Object` to an `&Object`

trait Foo {
    fn foo(&self) -> uint;
    fn bar(&mut self) -> uint;
}

impl Foo for uint {
    fn foo(&self) -> uint {
        *self
    }

    fn bar(&mut self) -> uint {
        *self += 1;
        *self
    }
}

fn do_it_mut(obj: &mut Foo) {
    let x = obj.bar();
    let y = obj.foo();
    assert_eq!(x, y);
}

fn do_it_imm(obj: &Foo, v: uint) {
    let y = obj.foo();
    assert_eq!(v, y);
}

fn main() {
    let x = @mut 22u as @mut Foo;
    do_it_mut(x);
    do_it_imm(x, 23u);
}
