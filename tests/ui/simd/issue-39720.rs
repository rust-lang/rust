//@ run-pass

#![feature(repr_simd, core_intrinsics)]

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct Char3([i8; 3]);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct Short3([i16; 3]);

fn main() {
    let cast: Short3 = unsafe { std::intrinsics::simd::simd_cast(Char3([10, -3, -9])) };

    // It's unclear what, if anything, this test was testing because #137108
    // showed that projecting into non-power-of-two types (as this was
    // originally doing) just wasn't working despite it.
    let cast = unsafe { &*std::ptr::from_ref(&cast).cast::<[i16; 3]>() };
    println!("{:?}", cast);
}
