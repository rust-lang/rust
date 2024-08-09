// Check that we report all error, since once cast may be intentional but another not,
// especially considering that `&T` to `&UnsafeCell<T>` may be valid but to `&mut T` never is.

use std::cell::UnsafeCell;
use std::mem::transmute;

#[repr(C)]
struct Foo(&'static u8, &'static u8);
#[repr(C)]
struct Bar(&'static UnsafeCell<u8>, &'static mut u8);

fn main() {
    let _a: Bar = unsafe { transmute(Foo(&0, &0)) };
    //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
    //~| ERROR transmuting &T to &mut T is undefined behavior, even if the reference is unused, consider instead using an UnsafeCell
}
