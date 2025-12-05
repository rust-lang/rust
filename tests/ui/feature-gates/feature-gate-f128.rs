//@ revisions: e2015 e2018
//
//@[e2018] edition:2018

#![allow(unused)]

const A: f128 = 10.0; //~ ERROR the type `f128` is unstable

pub fn main() {
    let a: f128 = 100.0; //~ ERROR the type `f128` is unstable
    let b = 0.0f128; //~ ERROR the type `f128` is unstable
    let c = 0f128; //~ ERROR the type `f128` is unstable
    foo(1.23);
}

fn foo(a: f128) {} //~ ERROR the type `f128` is unstable

struct Bar {
    a: f128, //~ ERROR the type `f128` is unstable
}
