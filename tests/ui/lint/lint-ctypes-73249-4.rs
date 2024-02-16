//@ check-pass
#![deny(improper_ctypes)]

use std::marker::PhantomData;

trait Foo {
    type Assoc;
}

impl Foo for () {
    type Assoc = PhantomData<()>;
}

#[repr(transparent)]
struct Wow<T> where T: Foo<Assoc = PhantomData<T>> {
    x: <T as Foo>::Assoc,
    v: u32,
}

extern "C" {
    fn test(v: Wow<()>);
}

fn main() {}
