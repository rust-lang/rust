//@ run-pass

#![allow(unused_imports)]
#![allow(unused_must_use)]
//@ pretty-expanded FIXME #23616

use std::fmt;
use std::marker::PhantomData;

trait Encoder {
    type Error;
}

trait Encodable<S: Encoder> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error>;
}

impl<S: Encoder> Encodable<S> for i32 {
    fn encode(&self, _s: &mut S) -> Result<(), S::Error> {
        Ok(())
    }
}

struct JsonEncoder<'a>(PhantomData<&'a mut ()>);

impl Encoder for JsonEncoder<'_> {
    type Error = ();
}

fn encode_json<T: for<'r> Encodable<JsonEncoder<'r>>>(
    object: &T,
) -> Result<String, ()> {
    let s = String::new();
    {
        let mut encoder = JsonEncoder(PhantomData);
        object.encode(&mut encoder)?;
    }
    Ok(s)
}

struct Foo<T: for<'a> Encodable<JsonEncoder<'a>>> {
    v: T,
}

impl<T: for<'a> Encodable<JsonEncoder<'a>>> Drop for Foo<T> {
    fn drop(&mut self) {
        encode_json(&self.v);
    }
}

fn main() {
    let _ = Foo { v: 10 };
}
