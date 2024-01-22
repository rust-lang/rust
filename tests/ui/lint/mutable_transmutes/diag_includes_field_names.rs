use std::cell::UnsafeCell;
use std::mem::transmute;

#[repr(transparent)]
struct A {
    a: B,
}
#[repr(transparent)]
struct B {
    b: C,
}
#[repr(transparent)]
struct C {
    c: &'static D,
}
#[repr(transparent)]
struct D {
    d: UnsafeCell<u8>,
}

#[repr(transparent)]
struct E {
    e: F,
}
#[repr(transparent)]
struct F {
    f: &'static G,
}
#[repr(transparent)]
struct G {
    g: H,
}
#[repr(transparent)]
struct H {
    h: u8,
}

fn main() {
    let _: A = unsafe { transmute(&1u8) };
    //~^ ERROR transmuting &T to &UnsafeCell<T> is undefined behavior, even if the reference is unused, consider using UnsafeCell on the original data
    let _: A = unsafe { transmute(E { e: F { f: &G { g: H { h: 0 } } } }) };
    //~^ ERROR transmuting &T to &UnsafeCell<T> is undefined behavior, even if the reference is unused, consider using UnsafeCell on the original data
    let _: &'static UnsafeCell<u8> = unsafe { transmute(E { e: F { f: &G { g: H { h: 0 } } } }) };
    //~^ ERROR transmuting &T to &UnsafeCell<T> is undefined behavior, even if the reference is unused, consider using UnsafeCell on the original data
}
