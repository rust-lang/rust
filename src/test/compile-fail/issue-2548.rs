// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A test case for #2548.

struct foo {
    x: @mut int,


}

impl foo : Drop {
    fn finalize(&self) {
        io::println("Goodbye, World!");
        *self.x += 1;
    }
}

fn foo(x: @mut int) -> foo {
    foo { x: x }
}

fn main() {
    let x = @mut 0;

    {
        let mut res = foo(x);

        let mut v = ~[];
        v = move ~[(move res)] + v; //~ ERROR instantiating a type parameter with an incompatible type (needs `copy`, got `&static`, missing `copy`)
        assert (v.len() == 2);
    }

    assert *x == 1;
}
