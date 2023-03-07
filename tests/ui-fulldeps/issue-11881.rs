// run-pass

#![allow(unused_must_use)]
#![allow(dead_code)]
#![allow(unused_imports)]

use std::fmt;
use std::io::prelude::*;
use std::io::Cursor;
use std::slice;
use std::marker::PhantomData;

trait Encoder {
    type Error;
}

trait Encodable<S: Encoder> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error>;
}

struct JsonEncoder<'a>(PhantomData<&'a mut ()>);

impl Encoder for JsonEncoder<'_> {
    type Error = ();
}

struct AsJson<'a, T> {
    inner: &'a T,
}

impl<'a, T: for<'r> Encodable<JsonEncoder<'r>>> fmt::Display for AsJson<'a, T> {
    /// Encodes a json value into a string
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

fn as_json<T>(t: &T) -> AsJson<'_, T> {
    AsJson { inner: t }
}

struct OpaqueEncoder(Vec<u8>);

impl Encoder for OpaqueEncoder {
    type Error = ();
}


struct Foo {
    baz: bool,
}

impl<S: Encoder> Encodable<S> for Foo {
    fn encode(&self, _s: &mut S) -> Result<(), S::Error> {
        Ok(())
    }
}

struct Bar {
    froboz: usize,
}

impl<S: Encoder> Encodable<S> for Bar {
    fn encode(&self, _s: &mut S) -> Result<(), S::Error> {
        Ok(())
    }
}

enum WireProtocol {
    JSON,
    Opaque,
    // ...
}

fn encode_json<T: for<'a> Encodable<JsonEncoder<'a>>>(val: &T, wr: &mut Cursor<Vec<u8>>) {
    write!(wr, "{}", as_json(val));
}

fn encode_opaque<T: Encodable<OpaqueEncoder>>(val: &T, wr: Vec<u8>) {
    let mut encoder = OpaqueEncoder(wr);
    val.encode(&mut encoder);
}

pub fn main() {
    let target = Foo { baz: false };
    let proto = WireProtocol::JSON;
    match proto {
        WireProtocol::JSON => encode_json(&target, &mut Cursor::new(Vec::new())),
        WireProtocol::Opaque => encode_opaque(&target, Vec::new()),
    }
}
