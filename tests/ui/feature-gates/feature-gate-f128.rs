#![allow(unused)]

const A: f128 = 10.0; //~ ERROR the type `f128` is unstable

pub fn main() {
    let a: f128 = 100.0; //~ ERROR the type `f128` is unstable
    let b = 0.0f128; //~ ERROR the type `f128` is unstable
    foo(1.23);
}

fn foo(a: f128) {} //~ ERROR the type `f128` is unstable

struct Bar {
    a: f128, //~ ERROR the type `f128` is unstable
}
