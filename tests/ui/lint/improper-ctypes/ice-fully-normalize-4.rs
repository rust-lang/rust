//@ check-pass
#![deny(improper_ctypes)]

// Issue: https://github.com/rust-lang/rust/issues/73249
// "ICE: could not fully normalize"

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
