#![allow(static_mut_refs)]

const C1: &'static mut [usize] = &mut [];
//~^ ERROR: mutable references are not allowed

static mut S: usize = 3;
const C2: &'static mut usize = unsafe { &mut S };
//~^ ERROR: referencing statics in constants

fn main() {}
