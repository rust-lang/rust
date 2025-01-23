//! This test case is a `#![no_core]`-version of the MVCE presented in #129301.
//!
//! The function [`delay()`] is removed, as it is not necessary to trigger the
//! wrong behavior and would require some additional lang items.
#![feature(no_core, lang_items, intrinsics, rustc_attrs)]
#![no_core]
#![no_main]
#![allow(internal_features)]

use minicore::ptr;

#[no_mangle]
pub fn main() -> ! {
    let port_b = 0x25 as *mut u8; // the I/O-address of PORTB

    // a simple loop with some trivial instructions within. This loop label has
    // to be placed correctly before the `ptr::write_volatile()` (some LLVM ver-
    // sions did place it after the first loop instruction, causing unsoundness)
    loop {
        unsafe { ptr::write_volatile(port_b, 1) };
        unsafe { ptr::write_volatile(port_b, 2) };
    }
}

// FIXME: replace with proper minicore once available (#130693)
mod minicore {
    #[lang = "pointee_sized"]
    pub trait PointeeSized {}

    #[lang = "meta_sized"]
    pub trait MetaSized: PointeeSized {}

    #[lang = "sized"]
    pub trait Sized: MetaSized {}

    #[lang = "copy"]
    pub trait Copy {}
    impl Copy for u32 {}
    impl Copy for &u32 {}
    impl<T: PointeeSized> Copy for *mut T {}

    pub mod ptr {
        #[inline]
        #[rustc_diagnostic_item = "ptr_write_volatile"]
        pub unsafe fn write_volatile<T>(dst: *mut T, src: T) {
            #[rustc_intrinsic]
            pub unsafe fn volatile_store<T>(dst: *mut T, val: T);

            unsafe { volatile_store(dst, src) };
        }
    }
}
