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

#[unsafe_destructor]
impl Drop for foo {
    fn drop(&self) {
        unsafe {
            println("Goodbye, World!");
            *self.x += 1;
        }
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
        v = ~[(res)] + v; //~ failed to find an implementation of trait
        assert_eq!(v.len(), 2);
    }

    assert_eq!(*x, 1);
}
