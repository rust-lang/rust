use std::cell::UnsafeCell;
use std::mem::transmute;

fn main() {
    let _a: &UnsafeCell<u8> = unsafe { transmute(&1u8) };
    //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
}
