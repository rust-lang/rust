// ignore-windows: No libc on Windows
// compile-flags: -Zmiri-check-number-validity

#![feature(rustc_private)]

/// Test that pthread_condattr_destroy doesn't trigger a number validity error.
extern crate libc;

fn main() {
    unsafe {
        use core::mem::MaybeUninit;
        let mut attr = MaybeUninit::<libc::pthread_condattr_t>::uninit();

        let r = libc::pthread_condattr_init(attr.as_mut_ptr());
        assert_eq!(r, 0);

        let r = libc::pthread_condattr_destroy(attr.as_mut_ptr());
        assert_eq!(r, 0);
    }
}
