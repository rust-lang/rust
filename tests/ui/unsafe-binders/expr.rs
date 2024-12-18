#![feature(unsafe_binders)]
//~^ WARN the feature `unsafe_binders` is incomplete

use std::unsafe_binder::{wrap_binder, unwrap_binder};

fn main() {
    let x = 1;
    let binder: unsafe<'a> &'a i32 = wrap_binder!(x);
    //~^ ERROR unsafe binders are not yet implemented
    //~| ERROR unsafe binders are not yet implemented
    let rx = *unwrap_binder!(binder);
    //~^ ERROR unsafe binders are not yet implemented
}
