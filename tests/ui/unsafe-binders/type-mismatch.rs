#![feature(unsafe_binders)]
//~^ WARN the feature `unsafe_binders` is incomplete

fn main() {
    let x: unsafe<> i32 = 0;
    //~^ ERROR mismatched types
    let x: unsafe<'a> &'a i32 = &0;
    //~^ ERROR mismatched types
}
