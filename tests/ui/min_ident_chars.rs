//@aux-build:proc_macros.rs:proc-macro
#![allow(irrefutable_let_patterns, nonstandard_style, unused)]
#![warn(clippy::min_ident_chars)]

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

struct O {
    o: u32,
}

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
    let z = 1;
    let y = 1;
    let z = 1;
    // Implicitly disallowed idents
    let h = 1;
    let e = 2;
    let l = 3;
    let l = 4;
    let o = 6;
    // 2 len does not lint
    let hi = 0;
    // Lint
    let (h, o, w) = (1, 2, 3);
    for (a, (r, e)) in (0..1000).enumerate().enumerate() {}
    let you = Vec4 { x: 1, y: 2, z: 3, w: 4 };
    while let (d, o, _i, n, g) = (true, true, false, false, true) {}
    let today = true;
    // Ideally this wouldn't lint, but this would (likely) require global analysis, outta scope
    // of this lint regardless
    let o = 1;
    let o = O { o };

    for j in 0..1000 {}
    for _ in 0..10 {}

    // Do not lint code from external macros
    external! { for j in 0..1000 {} }
    // Do not lint code from procedural macros
    with_span! {
        span
        for j in 0..1000 {}
    }
}

fn b() {}
fn wrong_pythagoras(a: f32, b: f32) -> f32 {
    a * a + a * b
}
