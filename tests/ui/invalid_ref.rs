#![feature(plugin)]
#![plugin(clippy)]

#![allow(unused)]
#![feature(core_intrinsics)]

extern crate core;
use std::intrinsics::{init, uninit};

fn main() {
    let x = 1; 
    unsafe { 
        ref_to_zeroed_std(&x);
        ref_to_zeroed_core(&x);
        ref_to_zeroed_intr(&x);
        ref_to_uninit_std(&x);
        ref_to_uninit_core(&x);
        ref_to_uninit_intr(&x);
    }
}

unsafe fn ref_to_zeroed_std<T: ?Sized>(t: &T) {
    let ref_zero: &T = std::mem::zeroed();     // warning
}

unsafe fn ref_to_zeroed_core<T: ?Sized>(t: &T) {
    let ref_zero: &T = core::mem::zeroed();   // warning
}

unsafe fn ref_to_zeroed_intr<T: ?Sized>(t: &T) {
    let ref_zero: &T = std::intrinsics::init();   // warning
}

unsafe fn ref_to_uninit_std<T: ?Sized>(t: &T) {
    let ref_uninit: &T = std::mem::uninitialized();   // warning
}

unsafe fn ref_to_uninit_core<T: ?Sized>(t: &T) {
    let ref_uninit: &T = core::mem::uninitialized();   // warning
}

unsafe fn ref_to_uninit_intr<T: ?Sized>(t: &T) {
    let ref_uninit: &T = std::intrinsics::uninit();   // warning
}

