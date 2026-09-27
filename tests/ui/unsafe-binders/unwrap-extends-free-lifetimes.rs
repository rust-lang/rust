//@ compile-flags: -Znext-solver
//@ check-fail

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::unsafe_binder::{unwrap_binder, wrap_binder};

fn launder_empty<'x>(r: &'x String) -> &'static String {
    let b: unsafe<> &'x String = unsafe { wrap_binder!(r) };
    unsafe { unwrap_binder!(b) } //~ ERROR lifetime may not live long enough
}

fn launder_free<'x>(r: &'x String) -> &'static String {
    let b: unsafe<'a> (&'a u8, &'x String) = unsafe { wrap_binder!((&0, r)) };
    unsafe { unwrap_binder!(b).1 } //~ ERROR lifetime may not live long enough
}

fn main() {
    let (r1, r2);
    {
        let s = String::from("hello");
        r1 = launder_empty(&s);
        r2 = launder_free(&s);
    }
    println!("{r1} {r2}");
}
