// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure that destructors get run on slice literals
struct foo {
    x: @mut int,
}

#[unsafe_destructor]
impl Drop for foo {
    fn drop(&self) {
        *self.x += 1;
    }
}

fn foo(x: @mut int) -> foo {
    foo {
        x: x
    }
}

pub fn main() {
    let x = @mut 0;
    {
        let l = &[foo(x)];
        assert_eq!(*l[0].x, 0);
    }
    assert_eq!(*x, 1);
}
