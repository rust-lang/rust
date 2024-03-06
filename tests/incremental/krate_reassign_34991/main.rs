//@ aux-build:a.rs
//@ revisions:rpass1 rpass2

#![feature(rustc_attrs)]

#[cfg(rpass1)]
extern crate a;

#[cfg(rpass1)]
pub fn use_X() -> u32 {
    let x: a::X = 22;
    x as u32
}

#[cfg(rpass2)]
pub fn use_X() -> u32 {
    22
}

pub fn main() { }
