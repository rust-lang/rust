#![feature(type_alias_impl_trait)]

//@ known-bug: #99840
// this should not compile
//@ check-pass

type Alias = impl Sized;

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
// impl Yay for i32 {} // this already errors
// impl Yay for u32 {} // this also already errors

fn main() {}
