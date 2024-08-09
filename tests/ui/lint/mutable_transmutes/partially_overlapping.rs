// Check that transmuting only part of the type to `UnsafeCell` triggers the lint.

use std::cell::UnsafeCell;
use std::mem::transmute;

#[repr(C)]
struct A(u32);

#[repr(C)]
struct B(u16, UnsafeCell<u8>);

#[repr(C)]
struct C(u8, UnsafeCell<u8>);

#[repr(C)]
struct D(UnsafeCell<u32>);

#[repr(C, packed)]
struct E(u8, UnsafeCell<u16>, u8);

#[repr(C, packed)]
struct F(UnsafeCell<u16>, u16);

fn main() {
    let _: &B = unsafe { transmute(&A(0)) };
    //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
    let _: &D = unsafe { transmute(&C(0, UnsafeCell::new(0))) };
    //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
    let _: &F = unsafe { transmute(&E(0, UnsafeCell::new(0), 0)) };
    //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
}
