#![feature(dyn_star)]
#![allow(incomplete_features)]

use core::fmt::Debug;

fn main() {
    let i = 42;
    let dyn_i = i as dyn* Debug;
    dbg!(dyn_i);
}
