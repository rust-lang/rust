// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let p;
    let a = 42;
    p = &a; //~ NOTE borrow occurs here
}
//~^ ERROR `a` does not live long enough
//~| NOTE `a` dropped here while still borrowed
//~| NOTE values in a scope are dropped in the opposite order they are created
