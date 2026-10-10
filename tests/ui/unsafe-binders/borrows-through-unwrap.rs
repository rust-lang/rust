//@ compile-flags: -Znext-solver
//@ check-pass

// We should be able to borrowck through place projections through an `unwrap_binder!()`.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::unsafe_binder::unwrap_binder;

fn two_borrows(mut b: unsafe<> (u8, u8)) {
    unsafe {
        let x = &mut unwrap_binder!(b).0;
        let y = &unwrap_binder!(b).1;
        *x = *y;
    }
}

fn guarded_match(b: unsafe<> Option<u8>) -> u8 {
    unsafe {
        match unwrap_binder!(b) {
            Some(x) if x > 0 => x,
            _ => 0,
        }
    }
}

fn main() {}
