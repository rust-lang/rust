//@ check-pass

use std::cell::UnsafeCell;
use std::mem::transmute;

fn main() {
    // Allow mutable reference to mutable reference.
    let _a: &mut u8 = unsafe { transmute(&mut 0u8) };

    // Allow mutable reference to `UnsafeCell`.
    let _a: &mut UnsafeCell<u8> = unsafe { transmute(&mut 0u8) };
    let _a: &UnsafeCell<u8> = unsafe { transmute(&mut 0u8) };

    // Allow `UnsafeCell` to `UnsafeCell`.
    {
        #[repr(C)]
        struct A {
            a: u32,
            b: UnsafeCell<u32>,
        }

        #[repr(C)]
        struct B {
            a: u32,
            b: UnsafeCell<u32>,
        }

        #[repr(transparent)]
        struct AWrapper(A);

        let _a: &UnsafeCell<u8> = unsafe { transmute(&UnsafeCell::new(0u8)) };
        let _a: &B = unsafe { transmute(&A { a: 0, b: UnsafeCell::new(0) }) };
        let _a: &AWrapper = unsafe { transmute(&A { a: 0, b: UnsafeCell::new(0) }) };
    }
}
