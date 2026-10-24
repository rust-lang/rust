#![feature(unsafe_binders)]

fn main() {
    let x: unsafe<> i32 = 0;
    //~^ ERROR mismatched types
    let x: unsafe<'a> &'a i32 = &0;
    //~^ ERROR mismatched types
}
