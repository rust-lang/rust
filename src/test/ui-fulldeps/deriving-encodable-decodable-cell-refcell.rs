// run-pass

#![allow(unused_imports)]
// This briefly tests the capability of `Cell` and `RefCell` to implement the
// `Encodable` and `Decodable` traits via `#[derive(Encodable, Decodable)]`
#![feature(rustc_private)]

extern crate rustc_macros;
extern crate rustc_serialize;

use rustc_macros::{Decodable, Encodable};
use rustc_serialize::opaque;
use rustc_serialize::{Decodable, Encodable, Encoder};
use std::cell::{Cell, RefCell};

#[derive(Encodable, Decodable)]
struct A {
    baz: isize,
}

#[derive(Encodable, Decodable)]
struct B {
    foo: Cell<bool>,
    bar: RefCell<A>,
}

fn main() {
    let obj = B { foo: Cell::new(true), bar: RefCell::new(A { baz: 2 }) };

    let mut encoder = opaque::Encoder::new();
    obj.encode(&mut encoder);
    let data = encoder.finish().unwrap();

    let mut decoder = opaque::Decoder::new(&data, 0);
    let obj2 = B::decode(&mut decoder);

    assert_eq!(obj.foo.get(), obj2.foo.get());
    assert_eq!(obj.bar.borrow().baz, obj2.bar.borrow().baz);
}
