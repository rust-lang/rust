//@ check-pass

#![feature(min_generic_const_args, generic_const_items)]
#![expect(incomplete_features)]

// Regression test for #133066 where we would try to evaluate `<() as Foo>::ASSOC<_>` even
// though it contained inference variables, which would cause ICEs.

trait Foo {
    #[rustc_always_gca]
    const ASSOC<const N: u32>: u32;
}

impl Foo for () {
    const ASSOC<const N: u32>: u32 = core::direct_const_arg!(N);
}

fn bar<const N: u32, T: Foo<ASSOC<N> = 10>>() {}

fn main() {
    bar::<_, ()>();
}
