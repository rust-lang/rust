//! This test was supposed to test that unsizing an uninferred but
//! `Sized` type works and will let inference happen later.
//! The issue was that what instead happened was that the inference
//! variable got equated to the unsized type, and subsequently failed
//! the `Sized` check.
//!
//! This was a wart in our coercion code, so it was removed to be instead
//! implemented in the trait system.

//@ revisions: nofallback fallback

#![feature(never_type)]
#![cfg_attr(fallback, feature(never_type_fallback))]
#![allow(unreachable_code)]

use std::error::Error;
use std::mem;

fn raw_ptr_box<T>(t: T) -> *mut T {
    panic!()
}

fn foo(x: !) -> Box<dyn Error> {
    // Method resolution will generate new inference vars and relate them.
    // Thus fallback will not fall back to `!`, but `()` instead.
    Box::<_ /* ! */>::new(x)
    //~^ ERROR cannot be known at compilation time
}

fn foo_raw_ptr(x: !) -> *mut dyn Error {
    /* *mut $0 is coerced to *mut Error here */
    raw_ptr_box::<_ /* ! */>(x)
    //~^ ERROR cannot be known at compilation time
    //~| ERROR cannot be known at compilation time
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
    //~^ ERROR cannot be known at compilation time

    let mut first_iter = false;
    loop {
        if !first_iter {
            let y: Box<dyn Xyz>
                = /* Box<$0> is coerced to Box<Xyz> here */ Box::new(x.unwrap());
            //~^ ERROR cannot be known at compilation time
            //~| ERROR cannot be known at compilation time
        }

        x = Some(S);
        //~^ ERROR mismatched types
        first_iter = true;
    }

    let mut y: Option<S> = None;
    // assert types are equal
    mem::swap(&mut x, &mut y);
    //~^ ERROR cannot be known at compilation time
    //~| ERROR mismatched types
}

fn main() {}
