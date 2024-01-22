use std::cell::UnsafeCell;
use std::mem::transmute;

#[repr(C)]
struct Foo<T> {
    a: u32,
    b: Bar<T>,
}
#[repr(C)]
struct Bar<T>(Baz<T>);
#[repr(C)]
struct Baz<T>(T);

#[repr(C)]
struct Other(&'static u8, &'static u8);

fn main() {
    let _: Foo<&'static mut u8> = unsafe { transmute(Other(&1u8, &1u8)) };
    //~^ ERROR transmuting &T to &mut T is undefined behavior, even if the reference is unused, consider instead using an UnsafeCell
    let _: Foo<&'static UnsafeCell<u8>> = unsafe { transmute(Other(&1u8, &1u8)) };
    //~^ ERROR transmuting &T to &UnsafeCell<T> is undefined behavior, even if the reference is unused, consider using UnsafeCell on the original data
}
