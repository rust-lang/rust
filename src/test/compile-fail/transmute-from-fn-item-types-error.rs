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
    //~^ ERROR is zero-sized and can't be transmuted
    //~^^ NOTE cast with `as` to a pointer instead

    let p = mem::transmute(foo);
    //~^ ERROR is zero-sized and can't be transmuted
    //~^^ NOTE cast with `as` to a pointer instead

    let of = mem::transmute(main);
    //~^ ERROR is zero-sized and can't be transmuted
    //~^^ NOTE cast with `as` to a pointer instead

    (i, p, of)
}

unsafe fn bar() {
    // Error as usual if the resulting type is not pointer-sized.
    mem::transmute::<_, u8>(main);
    //~^ ERROR transmute called with differently sized types
    //~^^ NOTE transmuting between 0 bits and 8 bits

    mem::transmute::<_, *mut ()>(foo);
    //~^ ERROR is zero-sized and can't be transmuted
    //~^^ NOTE cast with `as` to a pointer instead

    mem::transmute::<_, fn()>(bar);
    //~^ ERROR is zero-sized and can't be transmuted
    //~^^ NOTE cast with `as` to a pointer instead

    // No error if a coercion would otherwise occur.
    mem::transmute::<fn(), usize>(main);
}

unsafe fn baz() {
    mem::transmute::<_, *mut ()>(Some(foo));
    //~^ ERROR is zero-sized and can't be transmuted
    //~^^ NOTE cast with `as` to a pointer instead

    mem::transmute::<_, fn()>(Some(bar));
    //~^ ERROR is zero-sized and can't be transmuted
    //~^^ NOTE cast with `as` to a pointer instead

    mem::transmute::<_, Option<fn()>>(Some(baz));
    //~^ ERROR is zero-sized and can't be transmuted
    //~^^ NOTE cast with `as` to a pointer instead

    // No error if a coercion would otherwise occur.
    mem::transmute::<Option<fn()>, usize>(Some(main));
}

fn main() {
    unsafe {
        foo();
        bar();
        baz();
    }
}
