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
    //~^ NOTE reference must be valid for the block
    let long;
    let mut short = 0;
    //~^ NOTE but borrowed value is only valid for the block suffix following statement 1
    long = borrow(&mut short);
    //~^ ERROR `short` does not live long enough
}
