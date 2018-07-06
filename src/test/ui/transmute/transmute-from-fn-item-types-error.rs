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

unsafe fn foo() -> (i8, *const (), Option<fn()>) {
    let i = mem::transmute(bar);
    //~^ WARN transmutation from a type with an unspecified layout
    //~| ERROR transmute called with types of different sizes


    let p = mem::transmute(foo);
    //~^ WARN transmutation from a type with an unspecified layout
    //~| ERROR can't transmute zero-sized type


    let of = mem::transmute(main);
    //~^ WARN transmutation from a type with an unspecified layout
    //~| ERROR can't transmute zero-sized type


    (i, p, of)
}

unsafe fn bar() {
    // Error as usual if the resulting type is not pointer-sized.
    mem::transmute::<_, u8>(main);
    //~^ WARN transmutation from a type with an unspecified layout
    //~| ERROR transmute called with types of different sizes


    mem::transmute::<_, *mut ()>(foo);
    //~^ WARN transmutation from a type with an unspecified layout
    //~| ERROR can't transmute zero-sized type


    mem::transmute::<_, fn()>(bar);
    //~^ WARN transmutation from a type with an unspecified layout
    //~| ERROR can't transmute zero-sized type


    // No error if a coercion would otherwise occur.
    mem::transmute::<fn(), usize>(main);
}

unsafe fn baz() {
    mem::transmute::<_, *mut ()>(Some(foo));
    //~^ WARN transmutation from a type with an unspecified layout
    //~| ERROR can't transmute zero-sized type


    mem::transmute::<_, fn()>(Some(bar));
    //~^ WARN transmutation from a type with an unspecified layout
    //~| ERROR can't transmute zero-sized type


    mem::transmute::<_, Option<fn()>>(Some(baz));
    //~^ WARN transmutation from a type with an unspecified layout
    //~| ERROR can't transmute zero-sized type


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
