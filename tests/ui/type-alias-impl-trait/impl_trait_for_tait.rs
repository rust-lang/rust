//@ compile-flags: --crate-type=lib
//@ check-pass

#![feature(type_alias_impl_trait)]
type Alias = impl Sized;

#[define_opaque(Alias)]
fn constrain() -> Alias {
    1i32
}

trait HideIt {
    type Assoc;
}

impl HideIt for () {
    type Assoc = Alias;
}

pub trait Yay {}

impl Yay for <() as HideIt>::Assoc {}
