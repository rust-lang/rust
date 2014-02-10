// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that assignments to an `&mut` pointer which is found in a
// borrowed (but otherwise non-aliasable) location is illegal.

struct S<'a> {
    pointer: &'a mut int
}

fn copy_borrowed_ptr<'a,'b>(p: &'a mut S<'b>) -> S<'b> {
    S { pointer: &mut *p.pointer }
    //~^ ERROR lifetime of `p` is too short to guarantee its contents can be safely reborrowed
}

fn main() {
    let mut x = 1;

    {
        let mut y = S { pointer: &mut x };
        let z = copy_borrowed_ptr(&mut y);
        *y.pointer += 1;
        *z.pointer += 1;
    }
}
