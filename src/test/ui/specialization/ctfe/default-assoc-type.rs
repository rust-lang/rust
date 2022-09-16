// ICE fixed by #101478.
//
// See that PR for more details.
//
// check-pass
#![feature(specialization)]
//~^ WARNING the feature `specialization` is incomplete and may not be safe to use

trait Foo {
    type Assoc: Trait;
}

impl<T> Foo for Vec<T> {
    default type Assoc = u32;
}

trait Trait {
    const ASSOC: usize;
}

impl Trait for u32 {
    const ASSOC: usize = 0;
}

fn foo() -> [u8; 0] {
    [0; <<Vec<u32> as Foo>::Assoc as Trait>::ASSOC]
    //~^ WARNING relying on the underlying type of an opaque type in the type system
    //~| WARNING this was previously accepted by the compiler
    //~| WARNING relying on the underlying type of an opaque type in the type system
    //~| WARNING this was previously accepted by the compiler
}

fn main() {}
