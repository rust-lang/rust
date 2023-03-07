// run-pass
#![allow(unused_imports)]
// Test that we are able to compile calls to associated fns like
// `decode()` where the bound on the `Self` parameter references a
// lifetime parameter of the trait. This example indicates why trait
// lifetime parameters must be early bound in the type of the
// associated item.

// pretty-expanded FIXME #23616

use std::marker;

pub enum Value<'v> {
    A(&'v str),
    B,
}

pub trait Decoder<'v> {
    fn read(&mut self) -> Value<'v>;
}

pub trait Decodable<'v, D: Decoder<'v>> {
    fn decode(d: &mut D) -> Self;
}

impl<'v, D: Decoder<'v>> Decodable<'v, D> for () {
    fn decode(d: &mut D) -> () {
        match d.read() {
            Value::A(..) => (),
            Value::B => Decodable::decode(d),
        }
    }
}

pub fn main() { }
