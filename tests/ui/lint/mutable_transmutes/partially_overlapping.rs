// This test checks that transmuting part of `&T` to part of `&mut T` errors.
// I don't think this is necessary or common, but this is what the current code does.

use std::cell::UnsafeCell;
use std::mem::transmute;

#[repr(C, packed)]
struct Foo<T>(u8, T);

#[repr(C, packed)]
struct Bar(&'static u8, u8);

fn main() {
    let _: Foo<&'static mut u8> = unsafe { transmute(Bar(&1u8, 0)) };
    //~^ ERROR transmuting &T to &mut T is undefined behavior, even if the reference is unused, consider instead using an UnsafeCell
    let _: Foo<&'static UnsafeCell<u8>> = unsafe { transmute(Bar(&1u8, 0)) };
    //~^ ERROR transmuting &T to &UnsafeCell<T> is undefined behavior, even if the reference is unused, consider using UnsafeCell on the original data
}
