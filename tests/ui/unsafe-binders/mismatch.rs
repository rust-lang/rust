#![feature(unsafe_binders)]
//~^ WARN the feature `unsafe_binders` is incomplete

use std::unsafe_binder::{wrap_binder, unwrap_binder};

fn a() {
    let _: unsafe<'a> &'a i32 = wrap_binder!(&());
    //~^ ERROR unsafe binder casts are not fully implemented
    //~| ERROR mismatched types
}

fn b() {
    let _: i32 = wrap_binder!(&());
    //~^ ERROR unsafe binder casts are not fully implemented
    //~| ERROR `wrap_binder!()` can only wrap into unsafe binder
}

fn c() {
    let y = 1;
    unwrap_binder!(y);
    //~^ ERROR unsafe binder casts are not fully implemented
    //~| ERROR expected unsafe binder, found integer as input
}

fn d() {
    let unknown = Default::default();
    unwrap_binder!(unknown);
    //~^ ERROR unsafe binder casts are not fully implemented
    // FIXME(unsafe_binders): This should report ambiguity once we've removed
    // the error above which taints the infcx.
}

fn e() {
    let x = wrap_binder!(&42);
    //~^ ERROR unsafe binder casts are not fully implemented
    // Currently, type inference doesn't flow backwards for unsafe binders.
    // It could, perhaps, but that may cause even more surprising corners.
    // FIXME(unsafe_binders): This should report ambiguity once we've removed
    // the error above which taints the infcx.
    let _: unsafe<'a> &'a i32 = x;
}

fn main() {}
