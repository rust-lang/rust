// check-fail
// known-bug

// This should pass, but seems to run into a TAIT issue.

#![feature(generic_associated_types)]
#![feature(type_alias_impl_trait)]

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
    fn foo<'s>() -> Self::InnerStream<'s> { todo!() }
}

fn main() {}
