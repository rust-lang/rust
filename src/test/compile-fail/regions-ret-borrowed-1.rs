// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Similar to regions-ret-borrowed.rs, but using a named lifetime.  At
// some point regions-ret-borrowed reported an error but this file did
// not, due to special hardcoding around the anonymous region.

fn with<R, F>(f: F) -> R where F: for<'a> FnOnce(&'a isize) -> R {
    f(&3)
}

fn return_it<'a>() -> &'a isize {
    //~^ NOTE the lifetime must be valid for the lifetime 'a as defined on the block
    with(|o| o)
        //~^ ERROR cannot infer an appropriate lifetime due to conflicting requirements
        //~| ERROR cannot infer an appropriate lifetime due to conflicting requirements
        //~| ERROR cannot infer an appropriate lifetime due to conflicting requirements
        //~| NOTE the lifetime cannot outlive the lifetime 'a as defined on the block
        //~| NOTE ...so that expression is assignable (expected &isize, found &'a isize)
        //~| NOTE ...so that reference does not outlive borrowed content
        //~| NOTE cannot infer an appropriate lifetime
}

fn main() {
    let x = return_it();
    println!("foo={}", *x);
}
