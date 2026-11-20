//@ check-pass

#![feature(type_alias_impl_trait)]

struct Send<T> {
    i: InnerSend<T>,
}

type InnerSend<T> = impl Sized;

#[define_opaque(InnerSend)]
fn constrain<T>() -> InnerSend<T> {
    ()
}

trait SendMustNotImplDrop {}

#[allow(drop_bounds)]
impl<T: Drop> SendMustNotImplDrop for T {}

impl<T> SendMustNotImplDrop for Send<T> {}

fn main() {}
