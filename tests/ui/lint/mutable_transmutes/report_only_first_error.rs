use std::cell::UnsafeCell;
use std::mem::transmute;

#[repr(C)]
struct Foo(&'static u8, &'static u8);
#[repr(C)]
struct Bar(&'static UnsafeCell<u8>, &'static mut u8);

fn main() {
    let _a: Bar = unsafe { transmute(Foo(&0, &0)) };
    //~^ ERROR transmuting &T to &UnsafeCell<T> is undefined behavior, even if the reference is unused, consider using UnsafeCell on the original data
}
