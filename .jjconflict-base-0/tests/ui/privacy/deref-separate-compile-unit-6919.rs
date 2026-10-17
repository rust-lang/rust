// https://github.com/rust-lang/rust/issues/6919
//@ run-pass
#![allow(unused_attributes)]
//@ aux-build:iss-6919.rs

extern crate issue6919_3;

pub fn main() {
    let _ = issue6919_3::D.k;
}
