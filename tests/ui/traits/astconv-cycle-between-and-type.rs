//@ run-pass
// Test that we are able to successfully compile a setup where a trait
// (`Trait1`) references a struct (`SomeType<u32>`) which in turn
// carries a predicate that references the trait (`u32 : Trait1`,
// substituted).


#![allow(dead_code)]

trait Trait1 : Trait2<SomeType<u32>> {
    fn dumb(&self) { }
}

trait Trait2<A> {
    fn dumber(&self, _: A) { }
}

struct SomeType<A>
    where A : Trait1
{
    a: A
}

impl Trait1 for u32 { }

impl Trait2<SomeType<u32>> for u32 { }

fn main() { }
