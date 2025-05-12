//@ check-pass

#![feature(unsafe_binders)]
//~^ WARN the feature `unsafe_binders` is incomplete

use std::unsafe_binder::{wrap_binder, unwrap_binder};

fn main() {
    unsafe {
    let x = 1;
        let binder: unsafe<'a> &'a i32 = wrap_binder!(&x);
        let rx = *unwrap_binder!(binder);
    }
}
