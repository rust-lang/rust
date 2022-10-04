// ICE fixed by #102657.
//
// See that PR for more details.
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
    //~^ ERROR unable to use constant with a hidden value in the type system
    //~| ERROR mismatched types
}

fn main() {}
