// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

unsafe fn foo() -> (isize, *const (), Option<fn()>) {
    let i = mem::transmute(bar);
    //~^ WARN is now zero-sized and has to be cast to a pointer before transmuting
    //~^^ WARN was previously accepted

    let p = mem::transmute(foo);
    //~^ WARN is now zero-sized and has to be cast to a pointer before transmuting
    //~^^ WARN was previously accepted

    let of = mem::transmute(main);
    //~^ WARN is now zero-sized and has to be cast to a pointer before transmuting
    //~^^ WARN was previously accepted

    (i, p, of)
}

unsafe fn bar() {
    mem::transmute::<_, *mut ()>(foo);
    //~^ WARN is now zero-sized and has to be cast to a pointer before transmuting
    //~^^ WARN was previously accepted

    mem::transmute::<_, fn()>(bar);
    //~^ WARN is now zero-sized and has to be cast to a pointer before transmuting
    //~^^ WARN was previously accepted

    // No error if a coercion would otherwise occur.
    mem::transmute::<fn(), usize>(main);

    // Error, still, if the resulting type is not pointer-sized.
    mem::transmute::<_, u8>(main);
    //~^ ERROR transmute called with differently sized types
}

fn main() {
    unsafe {
        foo();
        bar();
    }
}
