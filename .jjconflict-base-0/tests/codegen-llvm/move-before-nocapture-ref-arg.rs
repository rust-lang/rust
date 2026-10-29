// Verify that move before the call of the function with noalias, nocapture, readonly.
// #107436
//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

#[repr(C)]
pub struct ThreeSlices<'a>(&'a [u32], &'a [u32], &'a [u32]);

#[no_mangle]
pub fn sum_slices(val: ThreeSlices) -> u32 {
    // CHECK-NOT: memcpy
    let val = val;
    sum(&val)
}

#[no_mangle]
#[inline(never)]
pub fn sum(val: &ThreeSlices) -> u32 {
    val.0.iter().sum::<u32>() + val.1.iter().sum::<u32>() + val.2.iter().sum::<u32>()
}
