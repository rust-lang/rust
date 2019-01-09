#![allow(dead_code)]
#![feature(rustc_private)]
#![no_std]

extern crate serialize as rustc_serialize;

#[derive(RustcEncodable)]
struct Bar {
    x: u32,
}

#[derive(RustcDecodable)]
struct Baz {
    x: u32,
}

fn main() {
    Bar { x: 0 };
    Baz { x: 0 };
}
