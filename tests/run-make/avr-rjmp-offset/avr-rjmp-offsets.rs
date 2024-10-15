//! This test case is a `#![no_core]`-version of the MVCE presented in #129301.
//!
//! The function [`delay()`] is minimized and does not actually contain a loop
//! in order to remove the need for additional lang items.
#![feature(no_core, lang_items, intrinsics, rustc_attrs, asm_experimental_arch)]
#![no_core]
#![no_main]
#![allow(internal_features)]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}

use minicore::ptr;

#[no_mangle]
pub fn main() -> ! {
    let port_b = 0x25 as *mut u8; // the I/O-address of PORTB

    // a simple loop with some trivial instructions within. This loop label has
    // to be placed correctly before the `ptr::write_volatile()` (some LLVM ver-
    // sions did place it after the first loop instruction, causing unsoundness)
    loop {
        unsafe { ptr::write_volatile(port_b, 1) };
        delay(500_0000);
        unsafe { ptr::write_volatile(port_b, 2) };
        delay(500_0000);
    }
}

#[inline(never)]
#[no_mangle]
fn delay(_: u32) {
    unsafe { asm!("nop") };
}

// FIXME: replace with proper minicore once available (#130693)
mod minicore {
    #[lang = "sized"]
    pub trait Sized {}

    #[lang = "copy"]
    pub trait Copy {}
    impl Copy for u32 {}
    impl Copy for &u32 {}
    impl<T: ?Sized> Copy for *mut T {}

    pub mod ptr {
        #[inline]
        #[rustc_diagnostic_item = "ptr_write_volatile"]
        pub unsafe fn write_volatile<T>(dst: *mut T, src: T) {
            extern "rust-intrinsic" {
                #[rustc_nounwind]
                pub fn volatile_store<T>(dst: *mut T, val: T);
            }
            unsafe { volatile_store(dst, src) };
        }
    }
}
