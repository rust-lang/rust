// check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![feature(const_generics_defaults)]
#![allow(incomplete_features)]

struct N;

struct Foo<const N: usize = 1, T = N>(T);

impl Foo {
    fn new() -> Self {
        Foo(N)
    }
}

fn main() {
    let Foo::<1, N>(N) = Foo::new();
}
