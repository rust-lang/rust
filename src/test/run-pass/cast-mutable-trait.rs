// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

trait T {
    fn foo(@mut self);
}

struct S {
    unused: int
}

impl T for S {
    fn foo(@mut self) {
    }
}

fn bar(t: @mut T) {
    t.foo();
}

pub fn main() {
    let s = @mut S { unused: 0 };
    let s2 = s as @mut T;
    s2.foo();
    bar(s2);
    bar(s as @mut T);
}
