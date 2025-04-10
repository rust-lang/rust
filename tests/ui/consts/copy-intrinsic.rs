#![stable(feature = "dummy", since = "1.0.0")]

// ignore-tidy-linelength
#![feature(intrinsics, staged_api, rustc_attrs)]
use std::mem;

#[stable(feature = "dummy", since = "1.0.0")]
#[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.63.0")]
#[rustc_intrinsic]
const unsafe fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize) {
    unimplemented!()
}

#[stable(feature = "dummy", since = "1.0.0")]
#[rustc_const_stable(feature = "const_intrinsic_copy", since = "1.63.0")]
#[rustc_intrinsic]
const unsafe fn copy<T>(src: *const T, dst: *mut T, count: usize) {
    unimplemented!()
}

const COPY_ZERO: () = unsafe {
    // Since we are not copying anything, this should be allowed.
    let src = ();
    let mut dst = ();
    copy_nonoverlapping(&src as *const _ as *const u8, &mut dst as *mut _ as *mut u8, 0);
};

const COPY_OOB_1: () = unsafe {
    let mut x = 0i32;
    let dangle = (&mut x as *mut i32).wrapping_add(10);
    // Zero-sized copy is fine.
    copy_nonoverlapping(0x100 as *const i32, dangle, 0);
    // Non-zero-sized copy is not.
    copy_nonoverlapping(0x100 as *const i32, dangle, 1); //~ ERROR evaluation of constant value failed [E0080]
    //~| NOTE got 0x100[noalloc] which is a dangling pointer
};
const COPY_OOB_2: () = unsafe {
    let x = 0i32;
    let dangle = (&x as *const i32).wrapping_add(10);
    // Zero-sized copy is fine.
    copy_nonoverlapping(dangle, 0x100 as *mut i32, 0);
    // Non-zero-sized copy is not.
    copy_nonoverlapping(dangle, 0x100 as *mut i32, 1); //~ ERROR evaluation of constant value failed [E0080]
    //~| NOTE +0x28 which is at or beyond the end of the allocation
};

const COPY_SIZE_OVERFLOW: () = unsafe {
    let x = 0;
    let mut y = 0;
    copy(&x, &mut y, 1usize << (mem::size_of::<usize>() * 8 - 1)); //~ ERROR evaluation of constant value failed [E0080]
    //~| NOTE overflow computing total size of `copy`
};
const COPY_NONOVERLAPPING_SIZE_OVERFLOW: () = unsafe {
    let x = 0;
    let mut y = 0;
    copy_nonoverlapping(&x, &mut y, 1usize << (mem::size_of::<usize>() * 8 - 1)); //~ ERROR evaluation of constant value failed [E0080]
    //~| NOTE overflow computing total size of `copy_nonoverlapping`
};

fn main() {
}
