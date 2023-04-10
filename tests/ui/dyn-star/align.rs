// revisions: normal over_aligned

#![feature(dyn_star)]
//~^ WARN the feature `dyn_star` is incomplete and may not be safe to use and/or cause compiler crashes

use std::fmt::Debug;

#[cfg_attr(over_aligned,      repr(C, align(1024)))]
#[cfg_attr(not(over_aligned), repr(C))]
#[derive(Debug)]
struct AlignedUsize(usize);

fn main() {
    let x = AlignedUsize(12) as dyn* Debug;
    //~^ ERROR `AlignedUsize` needs to have the same ABI as a pointer
}
