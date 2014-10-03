// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::rc::Rc;

struct Foo {
    f: Rc<int>,
}

impl Drop for Foo {
//~^ ERROR the trait `core::kinds::Send` is not implemented for the type `Foo`
//~^^ NOTE cannot implement a destructor on a structure or enumeration that does not satisfy Send
    fn drop(&mut self) {
    }
}

struct Bar<'a> {
    f: &'a int,
}

impl<'a> Drop for Bar<'a> {
//~^ ERROR E0141
    fn drop(&mut self) {
    }
}

struct Baz {
    f: &'static int,
}

impl Drop for Baz {
    fn drop(&mut self) {
    }
}

fn main() { }
