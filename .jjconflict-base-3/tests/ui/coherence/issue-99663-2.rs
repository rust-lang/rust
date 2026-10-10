//@ check-pass

#![feature(type_alias_impl_trait)]

struct Outer<T: ?Sized> {
    i: InnerSend<T>,
}

type InnerSend<T: ?Sized> = impl Send;

#[define_opaque(InnerSend)]
fn constrain<T: ?Sized>() -> InnerSend<T> {
    ()
}

trait SendMustNotImplDrop {}

#[allow(drop_bounds)]
impl<T: ?Sized + Send + Drop> SendMustNotImplDrop for T {}

impl<T: ?Sized> SendMustNotImplDrop for Outer<T> {}

fn main() {}
