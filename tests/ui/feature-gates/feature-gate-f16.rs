//@ revisions: e2015 e2018
//
//@[e2018] edition:2018

#![allow(unused)]

const A: f16 = 10.0; //~ ERROR the type `f16` is unstable

pub fn main() {
    let a: f16 = 100.0; //~ ERROR the type `f16` is unstable
    let b = 0.0f16; //~ ERROR the type `f16` is unstable
    let c = 0f16; //~ ERROR the type `f16` is unstable
    let into_f32: f32 = 1.0f16.into();
    //~^ ERROR the type `f16` is unstable
    //~| ERROR use of unstable library feature `f32_from_f16`
    let into_f64: f64 = 1.0_f16.into(); //~ ERROR the type `f16` is unstable
    let into_f128: f128 = 1.0_f16.into();
    //~^ ERROR the type `f16` is unstable
    //~| ERROR the type `f128` is unstable
    let from_i8: f16 = 1_i8.into(); //~ ERROR the type `f16` is unstable
    let from_u8: f16 = 1_u8.into(); //~ ERROR the type `f16` is unstable
    let from_bool: f16 = true.into(); //~ ERROR the type `f16` is unstable
    foo(1.23);
}

fn foo(a: f16) {} //~ ERROR the type `f16` is unstable

struct Bar {
    a: f16, //~ ERROR the type `f16` is unstable
}
