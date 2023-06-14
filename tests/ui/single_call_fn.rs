//@aux-build:proc_macros.rs
#![allow(unused)]
#![warn(clippy::single_call_fn)]
#![no_main]

#[macro_use]
extern crate proc_macros;

// Do not lint since it's public
pub fn f() {}

fn g() {
    f();
}

fn c() {
    println!("really");
    println!("long");
    println!("function...");
}

fn d() {
    c();
}

fn a() {}

fn b() {
    a();

    external! {
        fn lol() {
            lol_inner();
        }
        fn lol_inner() {}
    }
    with_span! {
        span
        fn lol2() {
            lol2_inner();
        }
        fn lol2_inner() {}
    }
}

fn e() {
    b();
    b();
}
