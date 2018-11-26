// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![feature(never_type)]
#![allow(unreachable_code)]

use std::error::Error;
use std::mem;

fn raw_ptr_box<T>(t: T) -> *mut T {
    panic!()
}

fn foo(x: !) -> Box<dyn Error> {
    /* *mut $0 is coerced to Box<dyn Error> here */ Box::<_ /* ! */>::new(x)
}

fn foo_raw_ptr(x: !) -> *mut dyn Error {
    /* *mut $0 is coerced to *mut Error here */ raw_ptr_box::<_ /* ! */>(x)
}

fn no_coercion(d: *mut dyn Error) -> *mut dyn Error {
    /* an unsize coercion won't compile here, and it is indeed not used
       because there is nothing requiring the _ to be Sized */
    d as *mut _
}

trait Xyz {}
struct S;
struct T;
impl Xyz for S {}
impl Xyz for T {}

fn foo_no_never() {
    let mut x /* : Option<S> */ = None;
    let mut first_iter = false;
    loop {
        if !first_iter {
            let y: Box<dyn Xyz>
                = /* Box<$0> is coerced to Box<Xyz> here */ Box::new(x.unwrap());
        }

        x = Some(S);
        first_iter = true;
    }

    let mut y : Option<S> = None;
    // assert types are equal
    mem::swap(&mut x, &mut y);
}

fn main() {
}
