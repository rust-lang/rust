#![feature(unsafe_binders)]
//~^ WARN the feature `unsafe_binders` is incomplete

fn main() {
    let x: unsafe<'a> &'a ();
    //~^ ERROR unsafe binders are not yet implemented
}
