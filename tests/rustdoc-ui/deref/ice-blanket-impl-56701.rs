//@ check-pass
// This shouldn't cause a stack overflow when rustdoc is run
// https://github.com/rust-lang/rust/issues/56701

use std::ops::Deref;
use std::ops::DerefMut;

pub trait SimpleTrait {
    type SimpleT;
}

impl<Inner: SimpleTrait, Outer: Deref<Target = Inner>> SimpleTrait for Outer {
    type SimpleT = Inner::SimpleT;
}

pub trait AnotherTrait {
    type AnotherT;
}

impl<T, Simple: SimpleTrait<SimpleT = Vec<T>>> AnotherTrait for Simple {
    type AnotherT = T;
}

pub struct Unrelated<Inner, UnrelatedT: DerefMut<Target = Vec<Inner>>>(UnrelatedT);

impl<Inner, UnrelatedT: DerefMut<Target = Vec<Inner>>> Deref for Unrelated<Inner, UnrelatedT> {
    type Target = Vec<Inner>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}


pub fn main() { }
