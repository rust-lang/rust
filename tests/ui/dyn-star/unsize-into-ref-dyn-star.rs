#![feature(dyn_star)]
#![allow(incomplete_features)]

use std::fmt::Debug;

fn main() {
    let i = 42 as &dyn* Debug;
    //~^ ERROR non-primitive cast: `i32` as `&dyn* Debug`
}
