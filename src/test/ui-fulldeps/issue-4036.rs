// run-pass
// Issue #4036: Test for an issue that arose around fixing up type inference
// byproducts in vtable records.

// pretty-expanded FIXME #23616

#![feature(rustc_private)]

extern crate rustc_serialize;

use rustc_serialize::{json, Decodable};

pub fn main() {
    let json = json::from_str("[1]").unwrap();
    let mut decoder = json::Decoder::new(json);
    let _x: Vec<isize> = Decodable::decode(&mut decoder);
}
