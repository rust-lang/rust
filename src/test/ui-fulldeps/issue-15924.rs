// run-pass

#![allow(unused_imports)]
#![allow(unused_must_use)]
// pretty-expanded FIXME #23616
#![feature(rustc_private)]

extern crate rustc_serialize;

use rustc_serialize::json;
use rustc_serialize::{Encodable, Encoder};
use std::fmt;

struct Foo<T: for<'a> Encodable<json::Encoder<'a>>> {
    v: T,
}

impl<T: for<'a> Encodable<json::Encoder<'a>>> Drop for Foo<T> {
    fn drop(&mut self) {
        json::encode(&self.v);
    }
}

fn main() {
    let _ = Foo { v: 10 };
}
