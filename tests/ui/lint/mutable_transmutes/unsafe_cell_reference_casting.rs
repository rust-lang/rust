use std::cell::UnsafeCell;

#[repr(C)]
struct A {
    a: u64,
    b: u32,
    c: u32,
}

#[repr(C)]
struct B {
    a: u64,
    b: UnsafeCell<u32>,
    c: u32,
}

fn main() {
    let a = A { a: 0, b: 0, c: 0 };
    let _b = unsafe { &*(&a as *const A as *const B) };
    //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
}
