#![allow(unused_imports)]
// This briefly tests the capability of `Cell` and `RefCell` to implement the
// `Encodable` and `Decodable` traits via `#[derive(Encodable, Decodable)]`


#![feature(rustc_private)]

extern crate serialize;
use serialize as rustc_serialize;

use std::cell::{Cell, RefCell};
use serialize::{Encodable, Decodable};
use serialize::json;

#[derive(RustcEncodable, RustcDecodable)]
struct A {
    baz: isize
}

#[derive(RustcEncodable, RustcDecodable)]
struct B {
    foo: Cell<bool>,
    bar: RefCell<A>,
}

fn main() {
    let obj = B {
        foo: Cell::new(true),
        bar: RefCell::new( A { baz: 2 } )
    };
    let s = json::encode(&obj).unwrap();
    let obj2: B = json::decode(&s).unwrap();
    assert_eq!(obj.foo.get(), obj2.foo.get());
    assert_eq!(obj.bar.borrow().baz, obj2.bar.borrow().baz);
}
