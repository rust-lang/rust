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

fn with<R>(f: <'a>|x: &'a int| -> R) -> R {
    f(&3)
}

fn return_it<'a>() -> &'a int {
    with(|o| o) //~ ERROR mismatched types
        //~^ ERROR lifetime of return value does not outlive the function call
        //~^^ ERROR cannot infer
}

fn main() {
    let x = return_it();
    println!("foo={}", *x);
}
