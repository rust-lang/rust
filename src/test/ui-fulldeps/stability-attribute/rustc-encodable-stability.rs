// edition:2018
#![allow(deprecated)]
#![feature(rustc_private)]

extern crate rustc_serialize;

#[derive(RustcDecodable, RustcEncodable)]
struct ArbitraryTestType {

}

fn main() {}
