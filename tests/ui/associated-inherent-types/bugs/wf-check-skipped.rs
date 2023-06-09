// known-bug: #100041
// check-pass

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// FIXME(inherent_associated_types): This should fail.

struct Foo;

impl Foo {
    type Bar<T> = ();
}

fn main() -> Foo::Bar::<Vec<[u32]>> {}
