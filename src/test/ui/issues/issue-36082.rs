// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

// FIXME(#49821) -- No tip about using a let binding

use std::cell::RefCell;

fn main() {
    let mut r = 0;
    let s = 0;
    let x = RefCell::new((&mut r,s));

    let val: &_ = x.borrow().0;
    //[ast]~^ ERROR borrowed value does not live long enough [E0597]
    //[ast]~| NOTE temporary value dropped here while still borrowed
    //[ast]~| NOTE temporary value does not live long enough
    //[ast]~| NOTE consider using a `let` binding to increase its lifetime
    //[mir]~^^^^^ ERROR borrowed value does not live long enough [E0597]
    //[mir]~| NOTE temporary value does not live long enough
    //[mir]~| NOTE temporary value only lives until here
    println!("{}", val);
    //[mir]~^ borrow later used here
}
//[ast]~^ NOTE temporary value needs to live until here
