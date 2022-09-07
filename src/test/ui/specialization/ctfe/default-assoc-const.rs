// ICE fixed by #101478.
//
// See that PR for more details.
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
    //~^ ERROR unable to use constant with a hidden value in the type system
    //~| ERROR mismatched types
}

fn main() {}
