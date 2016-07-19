// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(transmute_from_fn_item_types)]

use std::mem;

unsafe fn foo() -> (isize, *const (), Option<fn()>) {
    let i = mem::transmute(bar);
    //~^ ERROR is now zero-sized and has to be cast to a pointer before transmuting
    //~^^ WARNING was previously accepted

    let p = mem::transmute(foo);
    //~^ ERROR is now zero-sized and has to be cast to a pointer before transmuting
    //~^^ WARNING was previously accepted

    let of = mem::transmute(main);
    //~^ ERROR is now zero-sized and has to be cast to a pointer before transmuting
    //~^^ WARNING was previously accepted

    (i, p, of)
}

unsafe fn bar() {
    mem::transmute::<_, *mut ()>(foo);
    //~^ ERROR is now zero-sized and has to be cast to a pointer before transmuting
    //~^^ WARNING was previously accepted

    mem::transmute::<_, fn()>(bar);
    //~^ ERROR is now zero-sized and has to be cast to a pointer before transmuting
    //~^^ WARNING was previously accepted

    // No error if a coercion would otherwise occur.
    mem::transmute::<fn(), usize>(main);
}

fn main() {
    unsafe {
        foo();
        bar();
    }
}
