// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;

fn main() {
    let mut r = 0;
    let s = 0;
    let x = RefCell::new((&mut r,s));

    let val: &_ = x.borrow().0;
    //~^ WARNING this temporary used to live longer - see issue #39283
    //~^^ ERROR borrowed value does not live long enough
    //~| temporary value dropped here while still borrowed
    //~| temporary value created here
    //~| consider using a `let` binding to increase its lifetime
    //~| before rustc 1.16, this temporary lived longer - see issue #39283
    println!("{}", val);
}
//~^ temporary value needs to live until here
