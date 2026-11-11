//@ check-pass

#![feature(impl_trait_in_assoc_type)]

pub trait Stream {
    type Item;
}

impl Stream for () {
    type Item = i32;
}

trait Yay<AdditionalValue> {
    type InnerStream<'s>: Stream<Item = i32> + 's;
    fn foo<'s>() -> Self::InnerStream<'s>;
}

impl<'a> Yay<&'a ()> for () {
    type InnerStream<'s> = impl Stream<Item = i32> + 's;
    fn foo<'s>() -> Self::InnerStream<'s> {
        ()
    }
}

fn main() {}
