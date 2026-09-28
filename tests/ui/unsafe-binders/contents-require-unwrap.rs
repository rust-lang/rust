//@ compile-flags: -Znext-solver
//@ check-fail

// It should not be possible to read the discriminant or otherwise destructure
// and unsafe binder with unwrapping. See #158839.

#![feature(unsafe_binders, variant_count)]
#![allow(incomplete_features, dead_code)]

use std::unsafe_binder::{unwrap_binder, wrap_binder};

fn destructure(b: unsafe<'a> (&'a u8, u8)) {
    let (_x, _y) = b;
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE expected
    //~| NOTE this expression has type
}

fn match_variant(b: unsafe<> Option<u8>) {
    match b {
        //~^ NOTE this expression has type
        //~| NOTE this expression has type
        Some(_) => {}
        //~^ ERROR mismatched types
        //~| NOTE expected
        //~| NOTE expected
        None => {}
        //~^ ERROR mismatched types
        //~| NOTE expected
        //~| NOTE expected
        //~| NOTE `None` is interpreted as
    }
}

fn if_let(b: unsafe<> Option<u8>) {
    if let Some(_) = b {}
    //~^ ERROR mismatched types
    //~| NOTE expected
    //~| NOTE expected
    //~| NOTE found
    //~| NOTE this expression has type
}

fn wildcard(b: unsafe<> u8) {
    match b {
        _ => {}
    }
    let _ = b;
}

fn variant_count() {
    let _ = std::mem::variant_count::<unsafe<> Option<u8>>();
}

fn unwrapped(b: unsafe<'a> (&'a u8, Option<u8>)) -> u8 {
    unsafe {
        let (x, y) = unwrap_binder!(b);
        let _ = std::mem::discriminant(&y);
        if let Some(y) = y { *x + y } else { *x }
    }
}

fn main() {
    let b: unsafe<> Option<u8> = unsafe { wrap_binder!(Some(1)) };
    let _ = std::mem::discriminant(&b);
}
