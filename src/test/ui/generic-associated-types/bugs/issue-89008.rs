// check-fail
// edition:2021
// known-bug

// This should pass, but seems to run into a TAIT bug.

#![feature(type_alias_impl_trait)]
#![feature(generic_associated_types)]

use std::future::Future;

trait Stream {
    type Item;
}

struct Empty<T>(T);
impl<T> Stream for Empty<T> {
    type Item = ();
}
fn empty<T>() -> Empty<T> {
    todo!()
}

trait X {
    type LineStream<'a, Repr>: Stream<Item = Repr> where Self: 'a;

    type LineStreamFut<'a,Repr>: Future<Output = Self::LineStream<'a, Repr>> where Self: 'a;

    fn line_stream<'a,Repr>(&'a self) -> Self::LineStreamFut<'a,Repr>;
}

struct Y;

impl X for Y {
    type LineStream<'a, Repr> = impl Stream<Item = Repr>;

    type LineStreamFut<'a, Repr> = impl Future<Output = Self::LineStream<'a, Repr>> ;

    fn line_stream<'a, Repr>(&'a self) -> Self::LineStreamFut<'a, Repr> {
        async {empty()}
    }
}

fn main() {}
