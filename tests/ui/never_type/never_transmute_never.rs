//@ check-pass

#![crate_type="lib"]

#![feature(never_type)]
#![allow(dead_code)]
#![allow(unreachable_code)]
#![allow(unused_variables)]

struct Foo;

pub fn f(x: !) -> ! {
    x
}

pub fn ub() {
    // This is completely undefined behaviour,
    // but we still want to make sure it compiles.
    let x: ! = unsafe {
        std::mem::transmute::<Foo, !>(Foo)
    };
    f(x)
}
