// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure that you cannot use generic types to return a region outside
// of its bound.  Here, in the `return_it()` fn, we call with() but
// with R bound to &isize from the return_it.  Meanwhile, with()
// provides a value that is only good within its own stack frame. This
// used to successfully compile because we failed to account for the
// fact that fn(x: &isize) rebound the region &.

fn with<R, F>(f: F) -> R where F: FnOnce(&isize) -> R {
    f(&3)
}

fn return_it<'a>() -> &'a isize {
    //~^ NOTE the lifetime must be valid for the lifetime 'a as defined on the block
    with(|o| o)
        //~^ ERROR cannot infer an appropriate lifetime due to conflicting requirements
        //~| ERROR cannot infer an appropriate lifetime due to conflicting requirements
        //~| ERROR cannot infer an appropriate lifetime due to conflicting requirements
        //~| NOTE cannot infer an appropriate lifetime
        //~| NOTE the lifetime cannot outlive the anonymous lifetime #1 defined on the block
        //~| NOTE ...so that expression is assignable (expected &isize, found &isize)
        //~| NOTE ...so that reference does not outlive borrowed content
}

fn main() {
    let x = return_it();
    println!("foo={}", *x);
}
