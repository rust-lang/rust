// run-pass

#![allow(dead_code)]
#![feature(rustc_private)]

extern crate rustc_serialize;

use rustc_serialize::{json, Decodable};

trait JD: Decodable<json::Decoder> {}

fn exec<T: JD>() {
    let doc = json::from_str("").unwrap();
    let mut decoder = json::Decoder::new(doc);
    let _v: T = Decodable::decode(&mut decoder);
    panic!()
}

pub fn main() {}
