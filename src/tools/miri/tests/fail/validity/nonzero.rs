// gets masked by optimizations
//@compile-flags: -Zmir-opt-level=0
#![feature(rustc_attrs)]
#![allow(unused_attributes)]

union Moo {
    x: u32,
    y: std::num::NonZeroU32,
}

fn main() {
    // Make sure that we detect this even when no function call is happening along the way
    let _x = Some(unsafe { Moo { x: 0 }.y }); //~ ERROR: encountered 0, but expected something greater or equal to 1
}
