// run-pass

#![feature(rustc_private)]
#![allow(dead_code)]

extern crate serialize;

#[derive(Encodable)]
//~^ WARNING derive(Encodable) is deprecated in favor of derive(RustcEncodable)
struct Test1;

fn main() { }
