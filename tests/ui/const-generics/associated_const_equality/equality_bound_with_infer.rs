#![feature(associated_const_equality, generic_const_items)]
#![expect(incomplete_features)]

// Regression test for #133066 where we would try to evaluate `<() as Foo>::ASSOC<_>` even
// though it contained inference variables, which would cause ICEs.

trait Foo {
    const ASSOC<const N: u32>: u32;
}

impl Foo for () {
    const ASSOC<const N: u32>: u32 = N;
}

fn bar<const N: u32, T: Foo<ASSOC<N> = 10>>() {}

fn main() {
    bar::<_, ()>();
    //~^ ERROR: type mismatch resolving `<() as Foo>::ASSOC<_> == 10`

    // FIXME(mgca):
    // FIXME(associated_const_equality):
    // This ought to start compiling once const items are aliases rather than bodies
}
