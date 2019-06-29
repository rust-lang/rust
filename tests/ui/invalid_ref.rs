#![allow(deprecated, unused)]
#![feature(core_intrinsics)]

extern crate core;
use std::intrinsics::init;

fn main() {
    let x = 1;
    unsafe {
        ref_to_zeroed_std(&x);
        ref_to_zeroed_core(&x);
        ref_to_zeroed_intr(&x);
        ref_to_uninit_std(&x);
        ref_to_uninit_core(&x);
        some_ref();
        std_zeroed_no_ref();
        core_zeroed_no_ref();
        intr_init_no_ref();
    }
}

unsafe fn ref_to_zeroed_std<T: ?Sized>(t: &T) {
    let ref_zero: &T = std::mem::zeroed(); // warning
}

unsafe fn ref_to_zeroed_core<T: ?Sized>(t: &T) {
    let ref_zero: &T = core::mem::zeroed(); // warning
}

unsafe fn ref_to_zeroed_intr<T: ?Sized>(t: &T) {
    let ref_zero: &T = std::intrinsics::init(); // warning
}

unsafe fn ref_to_uninit_std<T: ?Sized>(t: &T) {
    let ref_uninit: &T = std::mem::uninitialized(); // warning
}

unsafe fn ref_to_uninit_core<T: ?Sized>(t: &T) {
    let ref_uninit: &T = core::mem::uninitialized(); // warning
}

fn some_ref() {
    let some_ref = &1;
}

unsafe fn std_zeroed_no_ref() {
    let mem_zero: usize = std::mem::zeroed(); // no warning
}

unsafe fn core_zeroed_no_ref() {
    let mem_zero: usize = core::mem::zeroed(); // no warning
}

unsafe fn intr_init_no_ref() {
    let mem_zero: usize = std::intrinsics::init(); // no warning
}
