#![feature(rustc_attrs, pattern_types, pattern_type_macro)]
#![allow(unused_attributes)]

#[repr(transparent)]
struct NonZero(std::pat::pattern_type!(u32 is 1..=u32::MAX));

fn main() {
    // Make sure that we detect this even when no function call is happening along the way
    let _x = Some(unsafe { NonZero(std::mem::transmute(0_u32)) }); //~ ERROR: encountered 0, but expected something greater or equal to 1
}
