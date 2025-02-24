#![feature(unsafe_binders)]
//~^ WARN the feature `unsafe_binders` is incomplete

use std::unsafe_binder::{wrap_binder, unwrap_binder};

fn a() {
    let _: unsafe<'a> &'a i32 = wrap_binder!(&());
    //~^ ERROR mismatched types
}

fn b() {
    let _: i32 = wrap_binder!(&());
    //~^ ERROR `wrap_binder!()` can only wrap into unsafe binder
}

fn c() {
    let y = 1;
    unwrap_binder!(y);
    //~^ ERROR expected unsafe binder, found integer as input
}

fn d() {
    let unknown = Default::default();
    //~^ ERROR type annotations needed
    unwrap_binder!(unknown);
}

fn e() {
    let x = wrap_binder!(&42);
    //~^ ERROR type annotations needed
    // Currently, type inference doesn't flow backwards for unsafe binders.
    // It could, perhaps, but that may cause even more surprising corners.
    let _: unsafe<'a> &'a i32 = x;
}

fn main() {}
