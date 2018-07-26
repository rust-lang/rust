// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the `'a` in the where clause correctly links the region
// of the output to the region of the input.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

trait FnLike<A,R> {
    fn call(&self, arg: A) -> R;
}

fn call_repeatedly<F>(f: F)
    where F : for<'a> FnLike<&'a isize, &'a isize>
{
    // Result is stored: cannot re-assign `x`
    let mut x = 3;
    let y = f.call(&x);
    x = 5; //[ast]~ ERROR cannot assign
           //[mir]~^ ERROR cannot assign to `x` because it is borrowed

    // Result is not stored: can re-assign `x`
    let mut x = 3;
    f.call(&x);
    f.call(&x);
    f.call(&x);
    x = 5;
    drop(y);
}

fn main() {
}
