#![allow(unused)]

const A: f16 = 10.0; //~ ERROR the type `f16` is unstable

pub fn main() {
    let a: f16 = 100.0; //~ ERROR the type `f16` is unstable
    let b = 0.0f16; //~ ERROR the type `f16` is unstable
    foo(1.23);
}

fn foo(a: f16) {} //~ ERROR the type `f16` is unstable

struct Bar {
    a: f16, //~ ERROR the type `f16` is unstable
}
