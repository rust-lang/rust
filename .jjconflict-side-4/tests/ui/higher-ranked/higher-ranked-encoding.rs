//! Regression test for https://github.com/rust-lang/rust/issues/15924

//@ run-pass

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

// This function uses higher-ranked trait bounds, which previously caused ICE
fn encode_json<T: for<'r> Encodable<JsonEncoder<'r>>>(object: &T) -> Result<String, ()> {
    let s = String::new();
    {
        let mut encoder = JsonEncoder(PhantomData);
        object.encode(&mut encoder)?;
    }
    Ok(s)
}

// Structure with HRTB constraint that was problematic
struct Foo<T: for<'a> Encodable<JsonEncoder<'a>>> {
    v: T,
}

// Drop implementation that exercises the HRTB bounds
impl<T: for<'a> Encodable<JsonEncoder<'a>>> Drop for Foo<T> {
    fn drop(&mut self) {
        let _ = encode_json(&self.v);
    }
}

fn main() {
    let _ = Foo { v: 10 };
}
