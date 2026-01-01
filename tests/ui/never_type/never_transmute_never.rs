//@ check-pass

#![feature(never_type)]
#![allow(dead_code)]
#![expect(unreachable_code)]
#![expect(unused_variables)]

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

fn main() {}
