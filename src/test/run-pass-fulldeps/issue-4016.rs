#![allow(dead_code)]

#![feature(rustc_private)]

extern crate serialize;

use serialize::{json, Decodable};

trait JD : Decodable {}

fn exec<T: JD>() {
    let doc = json::from_str("").unwrap();
    let mut decoder = json::Decoder::new(doc);
    let _v: T = Decodable::decode(&mut decoder).unwrap();
    panic!()
}

pub fn main() {}
