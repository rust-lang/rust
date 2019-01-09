#![allow(unused_imports)]
#![allow(unused_must_use)]
// pretty-expanded FIXME #23616

#![feature(rustc_private)]

extern crate serialize;

use std::fmt;
use serialize::{Encoder, Encodable};
use serialize::json;

struct Foo<T: Encodable> {
    v: T,
}

impl<T: Encodable> Drop for Foo<T> {
    fn drop(&mut self) {
        json::encode(&self.v);
    }
}

fn main() {
    let _ = Foo { v: 10 };
}
