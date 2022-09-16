// ICE fixed by #101478.
//
// See that PR for more details.
//
// check-pass
#![feature(specialization)]
//~^ WARNING the feature `specialization` is incomplete and may not be safe to use

trait Foo {
    const ASSOC: usize;
}


impl Foo for u32 {
    default const ASSOC: usize = 0;
}

fn foo() -> [u8; 0] {
    [0; <u32 as Foo>::ASSOC]
    //~^ WARNING relying on the underlying type of an opaque type in the type system
    //~| WARNING this was previously accepted by the compiler
    //~| WARNING relying on the underlying type of an opaque type in the type system
    //~| WARNING this was previously accepted by the compiler
}

fn main() {}
