// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A<'a> {
    func: &'a fn() -> Option<int>
}

impl<'a> A<'a> {
    fn call(&self) -> Option<int> {
        (*self.func)()
    }
}

fn foo() -> Option<int> {
    None
}

fn create() -> A<'static> {
    A {
        func: &foo, //~ ERROR borrowed value does not live long enough
    }
}

fn main() {
    let a = create();
    a.call();
}
