#![feature(dyn_star)]
#![allow(incomplete_features)]

use std::fmt::Debug;

fn main() {
    let i = [1, 2, 3, 4] as dyn* Debug;
    //~^ ERROR `[i32; 4]` needs to have the same alignment and size as a pointer
    dbg!(i);
}
