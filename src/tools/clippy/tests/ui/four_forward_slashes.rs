//@aux-build:proc_macros.rs
#![feature(custom_inner_attributes)]
#![allow(unused)]
#![warn(clippy::four_forward_slashes)]
#![no_main]
#![rustfmt::skip]

#[macro_use]
extern crate proc_macros;

//// whoops
//~^ four_forward_slashes
fn a() {}

//// whoops
//~^ four_forward_slashes
#[allow(dead_code)]
fn b() {}

//// whoops
//// two borked comments!
//~^^ four_forward_slashes
#[track_caller]
fn c() {}

fn d() {}

#[test]
//// between attributes
//~^ four_forward_slashes
#[allow(dead_code)]
fn g() {}

    //// not very start of contents
    //~^ four_forward_slashes
fn h() {}

fn i() {
    //// don't lint me bozo
    todo!()
}

external! {
    //// don't lint me bozo
    fn e() {}
}

with_span! {
    span
    //// don't lint me bozo
    fn f() {}
}
