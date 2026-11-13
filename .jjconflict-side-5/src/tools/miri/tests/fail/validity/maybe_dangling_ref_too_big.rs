#![feature(maybe_dangling)]
use std::mem::{MaybeDangling, transmute};

fn main() {
    let _x: MaybeDangling<&i8> = unsafe { transmute(usize::MAX) };
    //~^ERROR: too close to the end of the address space
}
