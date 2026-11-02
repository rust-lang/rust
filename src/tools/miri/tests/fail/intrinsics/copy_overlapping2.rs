//! Regression test for `copy_nonoverlapping` when all data is uninit.
#![feature(core_intrinsics)]

fn main() {
    let mut uninit = std::mem::MaybeUninit::<[u8; 4]>::uninit();
    let p = uninit.as_mut_ptr().cast::<u8>();
    // Directly call intrinsic to avoid debug assertions in the `std::ptr` version.
    unsafe { std::intrinsics::copy_nonoverlapping(p, p.add(1), 3) }; //~ ERROR: `copy_nonoverlapping` called on overlapping ranges
    unsafe { std::hint::unreachable_unchecked() };
}
