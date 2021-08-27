// check-pass
#![feature(const_generics_defaults)]

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
