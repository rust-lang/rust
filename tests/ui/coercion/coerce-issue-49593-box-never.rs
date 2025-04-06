//@ revisions: nofallback fallback
//@[fallback] check-pass

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
    //[nofallback]~^ ERROR trait bound `(): std::error::Error` is not satisfied
}

fn foo_raw_ptr(x: !) -> *mut dyn Error {
    /* *mut $0 is coerced to *mut Error here */
    raw_ptr_box::<_ /* ! */>(x)
    //[nofallback]~^ ERROR trait bound `(): std::error::Error` is not satisfied
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

    let mut y: Option<S> = None;
    // assert types are equal
    mem::swap(&mut x, &mut y);
}

fn main() {}
