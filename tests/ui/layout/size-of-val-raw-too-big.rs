//~ ERROR values of the type `Example` are too big for the target architecture
//@ build-fail
//@ compile-flags: --crate-type lib
//@ only-32bit Layout computation rejects this layout for different reasons on 64-bit.

#![feature(core_intrinsics)]
#![allow(internal_features)]

// isize::MAX is fine, but with the padding for the unsized tail it is too big.
#[repr(C)]
pub struct Example([u8; isize::MAX as usize], [u16]);

// We guarantee that with length 0, `size_of_val_raw` (which calls the `size_of_val` intrinsic)
// is safe to call. The compiler aborts compilation if a length of 0 would overflow.
// So let's construct a case where length 0 just barely overflows, and ensure that
// does abort compilation.
pub fn check(x: *const Example) -> usize {
    unsafe { std::intrinsics::size_of_val(x) }
}
