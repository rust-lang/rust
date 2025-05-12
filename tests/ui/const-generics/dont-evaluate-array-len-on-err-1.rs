#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

// This tests that during error handling for the "trait not implemented" error
// we dont try to evaluate std::mem::size_of::<Self::Assoc> causing an ICE

struct Adt;

trait Foo {
    type Assoc;
    fn foo()
    where
        [Adt; std::mem::size_of::<Self::Assoc>()]: ,
    {
        <[Adt; std::mem::size_of::<Self::Assoc>()] as Foo>::bar()
        //~^ ERROR the trait bound
    }

    fn bar() {}
}

fn main() {}
