// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct Foo {
    foo: Option<i32>,
}

impl Foo {
    pub fn new(foo: Option<i32>, _: ()) -> Foo {
        Foo { foo }
    }

    pub fn map(self) -> Option<Foo> {
        self.foo.map(Foo::new)
    }
    //~^^ ERROR function is expected to take 1 argument, but it takes 2 arguments [E0593]
}

enum Qux {
    Bar(i32),
}

fn foo<F>(f: F)
where
    F: Fn(),
{
}

fn main() {
    foo(Qux::Bar);
}
//~^^ ERROR function is expected to take 0 arguments, but it takes 1 argument [E0593]
