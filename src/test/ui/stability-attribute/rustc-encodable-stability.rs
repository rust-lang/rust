// edition:2018
#![allow(deprecated)]
extern crate rustc_serialize;

#[derive(RustcDecodable, RustcEncodable)]
struct ArbitraryTestType(());

fn main() {}
