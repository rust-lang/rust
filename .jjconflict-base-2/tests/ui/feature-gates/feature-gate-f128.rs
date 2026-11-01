//@ revisions: e2015 e2018
//
//@[e2018] edition:2018

#![allow(unused)]

const A: f128 = 10.0; //~ ERROR the type `f128` is unstable

pub fn main() {
    let a: f128 = 100.0; //~ ERROR the type `f128` is unstable
    let b = 0.0f128; //~ ERROR the type `f128` is unstable
    let c = 0f128; //~ ERROR the type `f128` is unstable
    let from_i8: f128 = 1_i8.into();
    //~^ ERROR the type `f128` is unstable
    //~| ERROR use of unstable library feature `f128`
    let from_u8: f128 = 1_u8.into(); //~ ERROR the type `f128` is unstable
    let from_i16: f128 = 1_i16.into(); //~ ERROR the type `f128` is unstable
    let from_u16: f128 = 1_u16.into(); //~ ERROR the type `f128` is unstable
    let from_i32: f128 = 1_i32.into(); //~ ERROR the type `f128` is unstable
    let from_u32: f128 = 1_u32.into(); //~ ERROR the type `f128` is unstable
    let from_i64: f128 = 1_i64.into(); //~ ERROR the type `f128` is unstable
    let from_u64: f128 = 1_u64.into(); //~ ERROR the type `f128` is unstable
    let from_f16: f128 = 1.0_f16.into();
    //~^ ERROR the type `f128` is unstable
    //~| ERROR the type `f16` is unstable
    let from_f32: f128 = 1.0_f32.into(); //~ ERROR the type `f128` is unstable
    let from_f64: f128 = 1.0_f64.into(); //~ ERROR the type `f128` is unstable
    let from_bool: f128 = true.into(); //~ ERROR the type `f128` is unstable
    foo(1.23);
}

fn foo(a: f128) {} //~ ERROR the type `f128` is unstable

struct Bar {
    a: f128, //~ ERROR the type `f128` is unstable
}
