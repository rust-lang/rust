//@aux-build:proc_macros.rs
#![allow(nonstandard_style, unused)]
#![warn(clippy::single_letter_idents)]

extern crate proc_macros;
use proc_macros::external;
use proc_macros::with_span;

struct A {
    a: u32,
    i: u32,
    A: u32,
    I: u32,
}

struct B(u32);

struct i;

enum C {
    D,
    E,
    F,
    j,
}

struct Vec4 {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

struct AA<T, E>(T, E);

fn main() {
    // Allowed idents
    let w = 1;
    // Ok, not this one
    // let i = 1;
    let j = 1;
    let n = 1;
    let x = 1;
    let y = 1;
    let z = 1;

    for j in 0..1000 {}

    // Do not lint code from external macros
    external! { for j in 0..1000 {} }
    // Do not lint code from procedural macros
    with_span! {
        span
        for j in 0..1000 {}
    }
}

fn b() {}
fn owo() {}
