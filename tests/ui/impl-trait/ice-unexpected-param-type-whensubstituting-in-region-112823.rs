// test for ICE #112823
// Unexpected parameter Type(Repr) when substituting in region

#![feature(impl_trait_in_assoc_type)]

use std::future::Future;

trait Stream {}

trait X {
    type LineStream<'a, Repr>
    where
        Self: 'a;
    type LineStreamFut<'a, Repr>
    where
        Self: 'a;
}

struct Y;

impl X for Y {
    type LineStream<'c, 'd> = impl Stream;
    //~^ ERROR type `LineStream` has 0 type parameters but its trait declaration has 1 type parameter
    type LineStreamFut<'a, Repr> = impl Future<Output = Self::LineStream<'a, Repr>>;
    fn line_stream<'a, Repr>(&'a self) -> Self::LineStreamFut<'a, Repr> {}
    //~^ ERROR `()` is not a future
    //~^^ method `line_stream` is not a member of trait `X`
}

pub fn main() {}
