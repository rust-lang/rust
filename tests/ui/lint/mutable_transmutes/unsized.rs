use std::cell::UnsafeCell;

#[repr(C)]
struct A<T: ?Sized> {
    a: u32,
    b: T,
}

#[repr(C)]
struct B {
    a: UnsafeCell<u32>,
    b: [u32],
}

fn main() {
    let a = &A { a: 0, b: [0_u32, 0] } as &A<[u32]>;
    let _b = unsafe { &*(a as *const A<[u32]> as *const B) };
    //~^ ERROR transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior
}
