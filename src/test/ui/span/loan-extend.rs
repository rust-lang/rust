// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(conservative_impl_trait)]

// Helper creating a fake borrow, captured by the impl Trait.
fn borrow<'a, T>(_: &'a mut T) -> impl Copy { () }

fn main() {
    let long;
    let mut short = 0;
    long = borrow(&mut short);
    //~^ NOTE borrow occurs here
}
//~^ ERROR `short` does not live long enough
//~| NOTE `short` dropped here while still borrowed
//~| NOTE values in a scope are dropped in the opposite order they are created
